"""Unit tests for ETAFlow DemoOrder, state machine, and SimulationEngine."""

import unittest

from etaflow.simulation.engine import SimulationEngine
from etaflow.simulation.order import (
    DemoOrder,
    InvalidStateTransitionError,
    OrderStatus,
)


class TestSimulationOrder(unittest.TestCase):
    """Test suite covering DemoOrder creation, validation, and lifecycle transitions."""

    def setUp(self) -> None:
        """Create sample order fixture."""
        self.order = DemoOrder(
            order_id="ORD-TEST-101",
            restaurant_lat=12.9352,
            restaurant_lon=77.6245,
            customer_lat=12.9550,
            customer_lon=77.6400,
            order_hour=20,
            day_of_week=4,
            type_of_order="Meal",
            festival="No",
            city="Metropolitian",
            predicted_duration_minutes=22.5,
        )

    def test_order_creation_defaults(self) -> None:
        """Verify order initializes with proper defaults and status."""
        self.assertEqual(self.order.status, OrderStatus.ORDER_RECEIVED)
        self.assertEqual(self.order.predicted_duration_minutes, 22.5)
        self.assertIsNone(self.order.simulated_actual_duration_minutes)
        self.assertEqual(self.order.stage_progress_pct, 0.0)

    def test_invalid_category_rejected(self) -> None:
        """Verify invalid category raises ValueError."""
        with self.assertRaises(ValueError):
            DemoOrder(
                order_id="ORD-BAD",
                restaurant_lat=12.9,
                restaurant_lon=77.6,
                customer_lat=12.95,
                customer_lon=77.64,
                order_hour=12,
                day_of_week=2,
                type_of_order="Pizza",  # Not in allowed vocabulary
            )

    def test_valid_sequential_transitions(self) -> None:
        """Verify standard lifecycle transitions work in order."""
        self.order.transition_to(OrderStatus.RESTAURANT_PREPARING)
        self.assertEqual(self.order.status, OrderStatus.RESTAURANT_PREPARING)

        self.order.transition_to(OrderStatus.READY_FOR_PICKUP)
        self.assertEqual(self.order.status, OrderStatus.READY_FOR_PICKUP)

        self.order.transition_to(OrderStatus.OUT_FOR_DELIVERY)
        self.assertEqual(self.order.status, OrderStatus.OUT_FOR_DELIVERY)

        self.order.transition_to(OrderStatus.DELIVERED)
        self.assertEqual(self.order.status, OrderStatus.DELIVERED)

    def test_invalid_transition_rejected(self) -> None:
        """Verify illegal jumps across states raise InvalidStateTransitionError."""
        # Cannot jump from ORDER_RECEIVED directly to DELIVERED
        with self.assertRaises(InvalidStateTransitionError):
            self.order.transition_to(OrderStatus.DELIVERED)

        # Move to DELIVERED
        self.order.transition_to(OrderStatus.RESTAURANT_PREPARING)
        self.order.transition_to(OrderStatus.READY_FOR_PICKUP)
        self.order.transition_to(OrderStatus.OUT_FOR_DELIVERY)
        self.order.transition_to(OrderStatus.DELIVERED)

        # Cannot move backwards from DELIVERED to RESTAURANT_PREPARING
        with self.assertRaises(InvalidStateTransitionError):
            self.order.transition_to(OrderStatus.RESTAURANT_PREPARING)

    def test_cancellation_from_active_states(self) -> None:
        """Verify order can be cancelled while active, but not once delivered."""
        self.order.transition_to(OrderStatus.CANCELLED)
        self.assertEqual(self.order.status, OrderStatus.CANCELLED)

        # Cannot cancel from terminal CANCELLED
        with self.assertRaises(InvalidStateTransitionError):
            self.order.transition_to(OrderStatus.RESTAURANT_PREPARING)

    def test_delay_metrics_pending_simulation(self) -> None:
        """Verify delay metrics when simulation is not completed."""
        metrics = self.order.calculate_delay_metrics()
        self.assertFalse(metrics["has_completed_simulation"])
        self.assertIsNone(metrics["simulated_actual_duration_minutes"])
        self.assertEqual(metrics["sla_classification"], "Pending Simulation")

    def test_delay_metrics_completed_simulation(self) -> None:
        """Verify signed diff and SLA classifications for Early, On-Time, and Late."""
        # On-Time case (within +-3m)
        self.order.predicted_duration_minutes = 20.0
        self.order.simulated_actual_duration_minutes = 21.5
        m_on_time = self.order.calculate_delay_metrics()
        self.assertTrue(m_on_time["has_completed_simulation"])
        self.assertEqual(m_on_time["signed_difference_minutes"], 1.5)
        self.assertEqual(m_on_time["absolute_error_minutes"], 1.5)
        self.assertEqual(m_on_time["sla_classification"], "On-Time")

        # Late case (> +3m)
        self.order.simulated_actual_duration_minutes = 26.0
        m_late = self.order.calculate_delay_metrics()
        self.assertEqual(m_late["signed_difference_minutes"], 6.0)
        self.assertEqual(m_late["sla_classification"], "Late")

        # Early case (< -3m)
        self.order.simulated_actual_duration_minutes = 15.0
        m_early = self.order.calculate_delay_metrics()
        self.assertEqual(m_early["signed_difference_minutes"], -5.0)
        self.assertEqual(m_early["sla_classification"], "Early")


class TestSimulationEngine(unittest.TestCase):
    """Test suite covering SimulationEngine lifecycle, clock, speed, and determinism."""

    def setUp(self) -> None:
        """Create fresh engine with test orders."""
        self.engine = SimulationEngine(random_seed=123, speed_multiplier=1.0)
        self.order1 = DemoOrder(
            order_id="ORD-001",
            restaurant_lat=12.9352,
            restaurant_lon=77.6245,
            customer_lat=12.9550,
            customer_lon=77.6400,
            order_hour=20,
            day_of_week=4,
            predicted_duration_minutes=20.0,
        )
        self.order2 = DemoOrder(
            order_id="ORD-002",
            restaurant_lat=12.9784,
            restaurant_lon=77.6408,
            customer_lat=12.9912,
            customer_lon=77.6534,
            order_hour=16,
            day_of_week=6,
            predicted_duration_minutes=15.0,
        )
        self.engine.add_order(self.order1)
        self.engine.add_order(self.order2)

    def test_empty_fleet_summary(self) -> None:
        """Verify summary returns zeros when no orders exist."""
        empty_engine = SimulationEngine()
        summary = empty_engine.get_fleet_summary()
        self.assertEqual(summary["total_orders"], 0)
        self.assertEqual(summary["avg_predicted_duration"], 0.0)

    def test_start_pause_resume_reset(self) -> None:
        """Verify engine state flags during execution lifecycle."""
        self.assertFalse(self.engine.is_running)
        self.assertFalse(self.engine.is_paused)

        self.engine.start()
        self.assertTrue(self.engine.is_running)
        self.assertFalse(self.engine.is_paused)

        self.engine.pause()
        self.assertTrue(self.engine.is_running)
        self.assertTrue(self.engine.is_paused)

        self.engine.resume()
        self.assertTrue(self.engine.is_running)
        self.assertFalse(self.engine.is_paused)

        self.engine.reset()
        self.assertFalse(self.engine.is_running)
        self.assertFalse(self.engine.is_paused)
        self.assertEqual(self.engine.simulation_clock_seconds, 0.0)

    def test_tick_advances_simulation_clock(self) -> None:
        """Verify tick method updates clock by real_delta * speed_multiplier."""
        self.engine.set_speed_multiplier(2.0)
        self.engine.start()

        self.engine.tick(real_delta_seconds=5.0)
        self.assertEqual(self.engine.simulation_clock_seconds, 10.0)

        # Paused engine does not advance
        self.engine.pause()
        self.engine.tick(real_delta_seconds=5.0)
        self.assertEqual(self.engine.simulation_clock_seconds, 10.0)

    def test_independent_order_progression(self) -> None:
        """Verify multiple orders advance progress and reach delivered state."""
        self.engine.start()
        # Large enough tick to progress through full delivery
        self.engine.tick(real_delta_seconds=200.0)

        self.assertEqual(self.order1.status, OrderStatus.DELIVERED)
        self.assertEqual(self.order2.status, OrderStatus.DELIVERED)
        self.assertIsNotNone(self.order1.simulated_actual_duration_minutes)
        self.assertIsNotNone(self.order2.simulated_actual_duration_minutes)

        summary = self.engine.get_fleet_summary()
        self.assertEqual(summary["total_orders"], 2)
        self.assertEqual(summary["completed"], 2)
        self.assertGreater(summary["avg_simulated_duration"], 0.0)

    def test_deterministic_reproducibility(self) -> None:
        """Verify that identical random seed produces identical simulated outcomes."""
        eng_a = SimulationEngine(random_seed=999)
        o_a = DemoOrder(
            order_id="ORD-DET",
            restaurant_lat=12.9352,
            restaurant_lon=77.6245,
            customer_lat=12.9550,
            customer_lon=77.6400,
            order_hour=19,
            day_of_week=3,
            predicted_duration_minutes=25.0,
        )
        eng_a.add_order(o_a)
        eng_a.start()
        eng_a.tick(300.0)

        eng_b = SimulationEngine(random_seed=999)
        o_b = DemoOrder(
            order_id="ORD-DET",
            restaurant_lat=12.9352,
            restaurant_lon=77.6245,
            customer_lat=12.9550,
            customer_lon=77.6400,
            order_hour=19,
            day_of_week=3,
            predicted_duration_minutes=25.0,
        )
        eng_b.add_order(o_b)
        eng_b.start()
        eng_b.tick(300.0)

        self.assertEqual(o_a.simulated_actual_duration_minutes, o_b.simulated_actual_duration_minutes)
        self.assertEqual(o_a.delay_cause, o_b.delay_cause)

    def test_simulation_start_begins_clock_and_progression(self) -> None:
        """Verify start begins clock and allows tick to advance orders."""
        # Before start, tick does not advance clock or progress
        self.assertFalse(self.engine.is_running)
        self.engine.tick(real_delta_seconds=5.0)
        self.assertEqual(self.engine.simulation_clock_seconds, 0.0)
        self.assertEqual(self.order1.stage_progress_pct, 0.0)

        # After start, tick advances both clock and orders
        self.engine.start()
        self.assertTrue(self.engine.is_running)
        self.assertFalse(self.engine.is_paused)
        self.engine.tick(real_delta_seconds=3.0)
        self.assertEqual(self.engine.simulation_clock_seconds, 3.0)
        self.assertGreater(self.order1.stage_progress_pct, 0.0)

    def test_simulation_pause_stops_progression_and_resume_continues(self) -> None:
        """Verify pause freezes clock/order progress, and resume continues progression."""
        self.engine.start()
        self.engine.tick(real_delta_seconds=10.0)
        saved_clock = self.engine.simulation_clock_seconds
        saved_progress = self.order1.stage_progress_pct
        self.assertEqual(saved_clock, 10.0)

        # Pause freezes clock and progress
        self.engine.pause()
        self.assertTrue(self.engine.is_running)
        self.assertTrue(self.engine.is_paused)
        self.engine.tick(real_delta_seconds=15.0)
        self.assertEqual(self.engine.simulation_clock_seconds, saved_clock)
        self.assertEqual(self.order1.stage_progress_pct, saved_progress)

        # Resume continues from the exact paused state
        self.engine.resume()
        self.assertTrue(self.engine.is_running)
        self.assertFalse(self.engine.is_paused)
        self.engine.tick(real_delta_seconds=5.0)
        self.assertEqual(self.engine.simulation_clock_seconds, saved_clock + 5.0)
        self.assertGreater(self.order1.stage_progress_pct, saved_progress)

    def test_simulation_reset_restores_initial_state(self) -> None:
        """Verify reset restores clock, orders, progress, and engine flags to pristine initial state."""
        self.engine.start()
        self.engine.tick(real_delta_seconds=250.0)
        self.assertEqual(self.order1.status, OrderStatus.DELIVERED)
        self.assertGreater(self.engine.simulation_clock_seconds, 0.0)

        self.engine.reset()
        self.assertFalse(self.engine.is_running)
        self.assertFalse(self.engine.is_paused)
        self.assertEqual(self.engine.simulation_clock_seconds, 0.0)
        self.assertEqual(self.order1.status, OrderStatus.ORDER_RECEIVED)
        self.assertEqual(self.order1.simulation_elapsed_seconds, 0.0)
        self.assertEqual(self.order1.stage_progress_pct, 0.0)
        self.assertIsNone(self.order1.simulated_actual_duration_minutes)
        self.assertIsNone(self.order1.delay_cause)

    def test_simulation_speed_multiplier_scaling(self) -> None:
        """Verify 1x, 2x, 5x, and 10x speed multipliers scale clock progression proportionally."""
        for speed in [1.0, 2.0, 5.0, 10.0]:
            eng = SimulationEngine(speed_multiplier=speed)
            eng.start()
            eng.tick(real_delta_seconds=2.0)
            self.assertAlmostEqual(eng.simulation_clock_seconds, 2.0 * speed, places=4)

    def test_simulation_order_lifecycle_all_five_stages_progression(self) -> None:
        """Verify order smoothly advances through all 5 sequential stages during simulation.

        Stages:
        1. ORDER_RECEIVED (0% -> 9.9%)
        2. RESTAURANT_PREPARING (10% -> 39.9%)
        3. READY_FOR_PICKUP (40% -> 54.9%)
        4. OUT_FOR_DELIVERY (55% -> 99.9%)
        5. DELIVERED (100%)
        """
        eng = SimulationEngine(speed_multiplier=1.0)
        # target_seconds for 10 min order: max(15.0, 10.0 * 6.0) = 60.0s
        test_order = DemoOrder(
            order_id="ORD-LIFECYCLE-1",
            restaurant_lat=12.9352,
            restaurant_lon=77.6245,
            customer_lat=12.9550,
            customer_lon=77.6400,
            order_hour=14,
            day_of_week=2,
            predicted_duration_minutes=10.0,
        )
        eng.add_order(test_order)
        eng.start()

        # Initial state
        self.assertEqual(test_order.status, OrderStatus.ORDER_RECEIVED)
        self.assertEqual(test_order.stage_progress_pct, 0.0)

        # Stage 1: Received (< 10% progress, e.g. 5.0s / 60.0s = 8.33%)
        eng.tick(real_delta_seconds=5.0)
        self.assertEqual(test_order.status, OrderStatus.ORDER_RECEIVED)

        # Stage 2: Restaurant Preparing (>= 10% progress, e.g. at 6.0s = 10%)
        eng.tick(real_delta_seconds=1.5)
        self.assertEqual(test_order.status, OrderStatus.RESTAURANT_PREPARING)

        # Stage 3: Ready for Pickup (>= 40% progress, e.g. at 24.0s = 40%)
        eng.tick(real_delta_seconds=18.0)
        self.assertEqual(test_order.status, OrderStatus.READY_FOR_PICKUP)

        # Stage 4: Out for Delivery (>= 55% progress, e.g. at 33.0s = 55%)
        eng.tick(real_delta_seconds=9.0)
        self.assertEqual(test_order.status, OrderStatus.OUT_FOR_DELIVERY)

        # Stage 5: Delivered (100% progress, e.g. at 60.0s = 100%)
        eng.tick(real_delta_seconds=30.0)
        self.assertEqual(test_order.status, OrderStatus.DELIVERED)
        self.assertEqual(test_order.stage_progress_pct, 100.0)
        self.assertIsNotNone(test_order.simulated_actual_duration_minutes)
        self.assertIsNotNone(test_order.delay_cause)


if __name__ == "__main__":
    unittest.main()

