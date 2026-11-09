"""Data contracts and state machine for simulated demo food delivery orders."""

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional, Set

from etaflow.api.schema import (
    VALID_CITIES,
    VALID_FESTIVALS,
    VALID_TYPES_OF_ORDER,
)


class OrderStatus(str, Enum):
    """Lifecycle states for a simulated food delivery order."""

    ORDER_RECEIVED = "ORDER_RECEIVED"
    RESTAURANT_PREPARING = "RESTAURANT_PREPARING"
    READY_FOR_PICKUP = "READY_FOR_PICKUP"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"


# Allowed state transitions
VALID_TRANSITIONS: Dict[OrderStatus, Set[OrderStatus]] = {
    OrderStatus.ORDER_RECEIVED: {OrderStatus.RESTAURANT_PREPARING, OrderStatus.CANCELLED},
    OrderStatus.RESTAURANT_PREPARING: {OrderStatus.READY_FOR_PICKUP, OrderStatus.CANCELLED},
    OrderStatus.READY_FOR_PICKUP: {OrderStatus.OUT_FOR_DELIVERY, OrderStatus.CANCELLED},
    OrderStatus.OUT_FOR_DELIVERY: {OrderStatus.DELIVERED, OrderStatus.CANCELLED},
    OrderStatus.DELIVERED: set(),
    OrderStatus.CANCELLED: set(),
}

# Standard SLA threshold in minutes for delay classification
SLA_ON_TIME_THRESHOLD_MINUTES: float = 3.0


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal order lifecycle transition is attempted."""


@dataclass
class DemoOrder:
    """Represents a simulated delivery order with geospatial anchors, prediction, and lifecycle tracking."""

    order_id: str
    restaurant_lat: float
    restaurant_lon: float
    customer_lat: float
    customer_lon: float
    order_hour: int
    day_of_week: int
    type_of_order: str = "Meal"
    festival: str = "No"
    city: str = "Metropolitian"
    predicted_duration_minutes: float = 20.0
    distance_km: float = 0.0
    status: OrderStatus = OrderStatus.ORDER_RECEIVED
    created_at: float = field(default_factory=time.time)
    simulation_elapsed_seconds: float = 0.0
    simulated_actual_duration_minutes: Optional[float] = None
    delay_cause: Optional[str] = None
    stage_progress_pct: float = 0.0

    def __post_init__(self) -> None:
        """Validate bounds and enum types on initialization."""
        if isinstance(self.status, str):
            self.status = OrderStatus(self.status)

        if self.type_of_order not in VALID_TYPES_OF_ORDER:
            raise ValueError(
                f"type_of_order '{self.type_of_order}' is invalid. Allowed: {sorted(VALID_TYPES_OF_ORDER)}."
            )
        if self.festival not in VALID_FESTIVALS:
            raise ValueError(
                f"festival '{self.festival}' is invalid. Allowed: {sorted(VALID_FESTIVALS)}."
            )
        if self.city not in VALID_CITIES:
            raise ValueError(
                f"city '{self.city}' is invalid. Allowed: {sorted(VALID_CITIES)}."
            )

    def can_transition_to(self, new_status: OrderStatus) -> bool:
        """Check whether transition to target status is valid from current state."""
        return new_status in VALID_TRANSITIONS.get(self.status, set())

    def transition_to(self, new_status: OrderStatus) -> None:
        """Transition order to new state, rejecting invalid lifecycle jumps."""
        if isinstance(new_status, str):
            new_status = OrderStatus(new_status)

        if not self.can_transition_to(new_status):
            raise InvalidStateTransitionError(
                f"Cannot transition order '{self.order_id}' from {self.status.value} to {new_status.value}."
            )
        self.status = new_status

    def calculate_delay_metrics(self) -> Dict[str, Any]:
        """Calculate signed difference, absolute error, and SLA delay classification."""
        if self.simulated_actual_duration_minutes is None:
            return {
                "has_completed_simulation": False,
                "predicted_duration_minutes": round(self.predicted_duration_minutes, 2),
                "simulated_actual_duration_minutes": None,
                "signed_difference_minutes": None,
                "absolute_error_minutes": None,
                "sla_classification": "Pending Simulation",
            }

        signed_diff = self.simulated_actual_duration_minutes - self.predicted_duration_minutes
        abs_err = abs(signed_diff)

        if signed_diff > SLA_ON_TIME_THRESHOLD_MINUTES:
            classification = "Late"
        elif signed_diff < -SLA_ON_TIME_THRESHOLD_MINUTES:
            classification = "Early"
        else:
            classification = "On-Time"

        return {
            "has_completed_simulation": True,
            "predicted_duration_minutes": round(self.predicted_duration_minutes, 2),
            "simulated_actual_duration_minutes": round(self.simulated_actual_duration_minutes, 2),
            "signed_difference_minutes": round(signed_diff, 2),
            "absolute_error_minutes": round(abs_err, 2),
            "sla_classification": classification,
            "delay_cause": self.delay_cause or "Normal operational variance",
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize demo order to JSON-friendly dictionary."""
        metrics = self.calculate_delay_metrics()
        return {
            "order_id": self.order_id,
            "restaurant_lat": round(self.restaurant_lat, 4),
            "restaurant_lon": round(self.restaurant_lon, 4),
            "customer_lat": round(self.customer_lat, 4),
            "customer_lon": round(self.customer_lon, 4),
            "order_hour": self.order_hour,
            "day_of_week": self.day_of_week,
            "type_of_order": self.type_of_order,
            "festival": self.festival,
            "city": self.city,
            "predicted_duration_minutes": round(self.predicted_duration_minutes, 2),
            "distance_km": round(self.distance_km, 2),
            "status": self.status.value,
            "simulation_elapsed_seconds": round(self.simulation_elapsed_seconds, 1),
            "stage_progress_pct": round(self.stage_progress_pct, 1),
            **metrics,
        }
