"""Simulation engine for managing multi-order demo fleet progression deterministically."""

import random
from typing import Any, Dict, List, Optional

from etaflow.features.pipeline import calculate_haversine_distance
from etaflow.simulation.order import (
    DemoOrder,
    InvalidStateTransitionError,
    OrderStatus,
)


class SimulationEngine:
    """Deterministic simulation engine driving order lifecycle and courier motion.

    NOTE: Methodological Transparency:
    The Random Forest model predicts total end-to-end delivery duration at booking time.
    Internal stage progress (kitchen prep, courier dispatch, road movement) represents
    simulated operational dynamics, not individual model sub-predictions.
    """

    def __init__(self, random_seed: int = 42, speed_multiplier: float = 1.0) -> None:
        """Initialize engine with specified random seed and initial speed multiplier."""
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        self.speed_multiplier = speed_multiplier
        self.is_running = False
        self.is_paused = False
        self.simulation_clock_seconds: float = 0.0
        self.orders: Dict[str, DemoOrder] = {}

    def set_speed_multiplier(self, speed: float) -> None:
        """Update simulation clock speed multiplier (e.g. 1.0, 2.0, 5.0, 10.0)."""
        if speed <= 0:
            raise ValueError("Speed multiplier must be positive.")
        self.speed_multiplier = float(speed)

    def add_order(self, order: DemoOrder) -> None:
        """Register a demo order into the active simulation fleet."""
        if order.distance_km <= 0:
            order.distance_km = calculate_haversine_distance(
                order.restaurant_lat,
                order.restaurant_lon,
                order.customer_lat,
                order.customer_lon,
            )
        self.orders[order.order_id] = order

    def get_order(self, order_id: str) -> Optional[DemoOrder]:
        """Fetch order by ID."""
        return self.orders.get(order_id)

    def cancel_order(self, order_id: str) -> None:
        """Cancel an in-flight order. Fails if order is already delivered."""
        order = self.orders.get(order_id)
        if not order:
            raise KeyError(f"Order '{order_id}' not found in fleet.")
        order.transition_to(OrderStatus.CANCELLED)

    def start(self) -> None:
        """Start or resume the simulation clock."""
        self.is_running = True
        self.is_paused = False

    def pause(self) -> None:
        """Pause the simulation clock."""
        if self.is_running:
            self.is_paused = True

    def resume(self) -> None:
        """Resume paused simulation."""
        if self.is_running and self.is_paused:
            self.is_paused = False

    def reset(self) -> None:
        """Reset simulation clock and return active orders to received state."""
        self.is_running = False
        self.is_paused = False
        self.simulation_clock_seconds = 0.0
        self.rng = random.Random(self.random_seed)

        for order in self.orders.values():
            order.status = OrderStatus.ORDER_RECEIVED
            order.simulation_elapsed_seconds = 0.0
            order.stage_progress_pct = 0.0
            order.simulated_actual_duration_minutes = None
            order.delay_cause = None

    def tick(self, real_delta_seconds: float) -> None:
        """Advance the simulation clock by real_delta_seconds * speed_multiplier.

        Updates stage progress and transitions order states smoothly.
        """
        if not self.is_running or self.is_paused:
            return

        effective_delta = real_delta_seconds * self.speed_multiplier
        self.simulation_clock_seconds += effective_delta

        for order in self.orders.values():
            if order.status in (OrderStatus.DELIVERED, OrderStatus.CANCELLED):
                continue

            order.simulation_elapsed_seconds += effective_delta

            # Target duration in simulation seconds (e.g., 20 mins -> scaled for demo time)
            # For interactive demo: 1 predicted minute = 6 simulation seconds (10x compressed scale)
            sim_target_seconds = max(15.0, order.predicted_duration_minutes * 6.0)
            progress = min(100.0, (order.simulation_elapsed_seconds / sim_target_seconds) * 100.0)
            order.stage_progress_pct = progress

            # Sequential state transitions as progress advances
            if progress >= 10.0 and order.status == OrderStatus.ORDER_RECEIVED:
                order.transition_to(OrderStatus.RESTAURANT_PREPARING)

            if progress >= 40.0 and order.status == OrderStatus.RESTAURANT_PREPARING:
                order.transition_to(OrderStatus.READY_FOR_PICKUP)

            if progress >= 55.0 and order.status == OrderStatus.READY_FOR_PICKUP:
                order.transition_to(OrderStatus.OUT_FOR_DELIVERY)

            if progress >= 100.0 and order.status == OrderStatus.OUT_FOR_DELIVERY:
                # Finalize simulated actual duration with deterministic operational variance
                order_seed = hash((self.random_seed, order.order_id)) & 0xFFFFFFFF
                sub_rng = random.Random(order_seed)
                variance_factor = sub_rng.uniform(-0.15, 0.25)  # -15% early to +25% late

                sim_actual = max(5.0, order.predicted_duration_minutes * (1.0 + variance_factor))
                order.simulated_actual_duration_minutes = sim_actual

                signed_err = sim_actual - order.predicted_duration_minutes
                if signed_err > 3.0:
                    causes = [
                        "Kitchen prep surge during peak dining hour",
                        "Simulated urban intersection road delay",
                        "Heavy festival order queue at merchant hub",
                    ]
                    order.delay_cause = sub_rng.choice(causes)
                elif signed_err < -3.0:
                    order.delay_cause = "Expedited kitchen handoff & smooth corridor transit"
                else:
                    order.delay_cause = "On-time arrival within SLA threshold"

                order.transition_to(OrderStatus.DELIVERED)

    def get_fleet_summary(self) -> Dict[str, Any]:
        """Aggregate operational metrics across all registered demo orders."""
        total = len(self.orders)
        if total == 0:
            return {
                "total_orders": 0,
                "awaiting_pickup": 0,
                "in_transit": 0,
                "completed": 0,
                "cancelled": 0,
                "late_orders": 0,
                "avg_predicted_duration": 0.0,
                "avg_simulated_duration": 0.0,
                "avg_absolute_difference": 0.0,
                "on_time_rate_pct": 0.0,
            }

        awaiting_pickup = sum(
            1 for o in self.orders.values()
            if o.status in (OrderStatus.ORDER_RECEIVED, OrderStatus.RESTAURANT_PREPARING, OrderStatus.READY_FOR_PICKUP)
        )
        in_transit = sum(1 for o in self.orders.values() if o.status == OrderStatus.OUT_FOR_DELIVERY)
        completed = sum(1 for o in self.orders.values() if o.status == OrderStatus.DELIVERED)
        cancelled = sum(1 for o in self.orders.values() if o.status == OrderStatus.CANCELLED)

        avg_pred = sum(o.predicted_duration_minutes for o in self.orders.values()) / total

        completed_orders = [o for o in self.orders.values() if o.status == OrderStatus.DELIVERED]
        if completed_orders:
            avg_sim = sum(o.simulated_actual_duration_minutes or 0.0 for o in completed_orders) / len(completed_orders)
            avg_err = sum(
                abs((o.simulated_actual_duration_minutes or 0.0) - o.predicted_duration_minutes)
                for o in completed_orders
            ) / len(completed_orders)
            late_count = sum(
                1 for o in completed_orders
                if ((o.simulated_actual_duration_minutes or 0.0) - o.predicted_duration_minutes) > 3.0
            )
            on_time_count = sum(
                1 for o in completed_orders
                if abs((o.simulated_actual_duration_minutes or 0.0) - o.predicted_duration_minutes) <= 3.0
            )
            on_time_rate = (on_time_count / len(completed_orders)) * 100.0
        else:
            avg_sim = 0.0
            avg_err = 0.0
            late_count = 0
            on_time_rate = 0.0

        return {
            "total_orders": total,
            "awaiting_pickup": awaiting_pickup,
            "in_transit": in_transit,
            "completed": completed,
            "cancelled": cancelled,
            "late_orders": late_count,
            "avg_predicted_duration": round(avg_pred, 2),
            "avg_simulated_duration": round(avg_sim, 2),
            "avg_absolute_difference": round(avg_err, 2),
            "on_time_rate_pct": round(on_time_rate, 1),
        }
