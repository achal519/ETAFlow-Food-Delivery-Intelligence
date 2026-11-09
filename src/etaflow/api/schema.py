"""Inference request and response data contracts for ETAFlow.

Enforces strict prediction-time feature contracts, rejecting post-dispatch features
and invalid coordinates, non-finite values, and invalid categories.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set, Tuple

# Allowed categories matching the training dataset vocabulary
VALID_TYPES_OF_ORDER: Set[str] = {"Buffet", "Drinks", "Meal", "Snack"}
VALID_FESTIVALS: Set[str] = {"No", "Yes"}
VALID_CITIES: Set[str] = {"Metropolitian", "Semi-Urban", "Urban"}

# Forbidden post-booking and leakage columns that must trigger an HTTP 400 rejection
FORBIDDEN_POST_BOOKING_FIELDS: Set[str] = {
    # Target and target-derived leakage
    "delivery_speed",
    "time_taken (min)",
    "time_taken",
    "delivery_duration_min",
    "delivery_duration",
    "actual_delivery_duration",
    "actual_time_taken",
    # Post-booking kitchen / pickup timestamps
    "time_order_picked",
    "pickup_time",
    "actual_pickup_time",
    "actual_transit_time",
    "rider_dropoff_notes",
    "delivery_status_timestamp",
    # Courier identifiers and post-dispatch attributes
    "delivery_person_id",
    "courier_id",
    "rider_id",
    "delivery_person_age",
    "courier_age",
    "delivery_person_ratings",
    "courier_ratings",
    "courier_rating",
    "rider_rating",
    "vehicle_condition",
    "type_of_vehicle",
    "multiple_deliveries",
}


def _is_finite_number(val: Any) -> bool:
    """Check if a value is a real finite number (not NaN or Inf)."""
    if not isinstance(val, (int, float)):
        return False
    return not (math.isnan(val) or math.isinf(val))


@dataclass
class ETAPredictionRequest:
    """Input payload contract for order ETA prediction at checkout/booking time.

    Accepts the 11 Tier 1 booking-time features (or raw coordinate / date inputs
    from which they are derived).
    """

    restaurant_lat: float
    restaurant_lon: float
    customer_lat: float
    customer_lon: float
    order_hour: int
    day_of_week: int
    type_of_order: str = "Meal"
    festival: str = "No"
    city: str = "Metropolitian"
    order_id: Optional[str] = None
    # Quarantined Tier 2 attributes preserved for backward-compatible interface
    weather_condition: Optional[str] = "Normal"
    traffic_density: Optional[str] = "Medium"

    def validate(self) -> List[str]:
        """Validate input payload constraints, types, and categorical bounds."""
        errors: List[str] = []

        # Validate coordinate finiteness and physical ranges
        if not _is_finite_number(self.restaurant_lat):
            errors.append("restaurant_lat must be a finite number.")
        elif not (-90.0 <= self.restaurant_lat <= 90.0):
            errors.append("restaurant_lat must be between -90 and 90.")

        if not _is_finite_number(self.restaurant_lon):
            errors.append("restaurant_lon must be a finite number.")
        elif not (-180.0 <= self.restaurant_lon <= 180.0):
            errors.append("restaurant_lon must be between -180 and 180.")

        if not _is_finite_number(self.customer_lat):
            errors.append("customer_lat must be a finite number.")
        elif not (-90.0 <= self.customer_lat <= 90.0):
            errors.append("customer_lat must be between -90 and 90.")

        if not _is_finite_number(self.customer_lon):
            errors.append("customer_lon must be a finite number.")
        elif not (-180.0 <= self.customer_lon <= 180.0):
            errors.append("customer_lon must be between -180 and 180.")

        # Validate timing integers
        if not isinstance(self.order_hour, int) or isinstance(self.order_hour, bool):
            errors.append("order_hour must be an integer.")
        elif not (0 <= self.order_hour <= 23):
            errors.append("order_hour must be between 0 and 23.")

        if not isinstance(self.day_of_week, int) or isinstance(self.day_of_week, bool):
            errors.append("day_of_week must be an integer.")
        elif not (0 <= self.day_of_week <= 6):
            errors.append("day_of_week must be between 0 (Mon) and 6 (Sun).")

        # Validate categorical domains
        if self.type_of_order not in VALID_TYPES_OF_ORDER:
            errors.append(
                f"type_of_order '{self.type_of_order}' is invalid. Allowed: {sorted(VALID_TYPES_OF_ORDER)}."
            )

        if self.festival not in VALID_FESTIVALS:
            errors.append(
                f"festival '{self.festival}' is invalid. Allowed: {sorted(VALID_FESTIVALS)}."
            )

        if self.city not in VALID_CITIES:
            errors.append(
                f"city '{self.city}' is invalid. Allowed: {sorted(VALID_CITIES)}."
            )

        return errors

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Tuple[Optional["ETAPredictionRequest"], List[str]]:
        """Parse and validate an incoming dictionary payload.

        Checks for forbidden post-booking fields, missing required fields,
        and invalid data types or bounds.
        """
        if not isinstance(data, dict):
            return None, ["Request body must be a JSON object."]

        # 1. Strictly detect forbidden post-booking fields
        forbidden_found: List[str] = []
        for key in data.keys():
            normalized = key.strip().lower()
            if normalized in FORBIDDEN_POST_BOOKING_FIELDS or any(
                f in normalized for f in [
                    "time_order_picked", "pickup_time", "delivery_speed",
                    "time_taken", "delivery_person", "courier", "rider_rating"
                ]
            ):
                forbidden_found.append(key)

        if forbidden_found:
            return None, [
                f"Forbidden post-booking field '{k}' is rejected at booking time."
                for k in sorted(forbidden_found)
            ]

        # 2. Check for required booking-time fields
        # Support both canonical API snake_case and dataframe PascalCase aliases
        def _get_field(aliases: List[str]) -> Tuple[Any, bool]:
            for a in aliases:
                if a in data:
                    return data[a], True
            return None, False

        restaurant_lat_val, rlat_found = _get_field(["restaurant_lat", "Restaurant_latitude"])
        restaurant_lon_val, rlon_found = _get_field(["restaurant_lon", "Restaurant_longitude"])
        customer_lat_val, clat_found = _get_field(["customer_lat", "Delivery_location_latitude"])
        customer_lon_val, clon_found = _get_field(["customer_lon", "Delivery_location_longitude"])
        order_hour_val, ohour_found = _get_field(["order_hour", "Time_Orderd_hour"])
        day_of_week_val, dow_found = _get_field(["day_of_week"])

        missing_fields: List[str] = []
        if not rlat_found:
            missing_fields.append("restaurant_lat")
        if not rlon_found:
            missing_fields.append("restaurant_lon")
        if not clat_found:
            missing_fields.append("customer_lat")
        if not clon_found:
            missing_fields.append("customer_lon")
        if not ohour_found:
            missing_fields.append("order_hour")
        if not dow_found:
            missing_fields.append("day_of_week")

        if missing_fields:
            return None, [f"Missing required booking field: {field_name}" for field_name in missing_fields]

        # 3. Type parsing
        errors: List[str] = []
        parsed_rlat = None
        parsed_rlon = None
        parsed_clat = None
        parsed_clon = None
        parsed_hour = None
        parsed_dow = None

        try:
            parsed_rlat = float(restaurant_lat_val)
        except (ValueError, TypeError):
            errors.append("restaurant_lat must be a valid float.")

        try:
            parsed_rlon = float(restaurant_lon_val)
        except (ValueError, TypeError):
            errors.append("restaurant_lon must be a valid float.")

        try:
            parsed_clat = float(customer_lat_val)
        except (ValueError, TypeError):
            errors.append("customer_lat must be a valid float.")

        try:
            parsed_clon = float(customer_lon_val)
        except (ValueError, TypeError):
            errors.append("customer_lon must be a valid float.")

        try:
            if isinstance(order_hour_val, bool):
                raise ValueError()
            parsed_hour = int(order_hour_val)
        except (ValueError, TypeError):
            errors.append("order_hour must be a valid integer.")

        try:
            if isinstance(day_of_week_val, bool):
                raise ValueError()
            parsed_dow = int(day_of_week_val)
        except (ValueError, TypeError):
            errors.append("day_of_week must be a valid integer.")

        if errors:
            return None, errors

        # 4. Optional / categorical fields
        type_of_order_val, _ = _get_field(["type_of_order", "Type_of_order"])
        festival_val, _ = _get_field(["festival", "Festival"])
        city_val, _ = _get_field(["city", "City"])
        order_id_val, _ = _get_field(["order_id"])

        req = cls(
            restaurant_lat=parsed_rlat,  # type: ignore[arg-type]
            restaurant_lon=parsed_rlon,  # type: ignore[arg-type]
            customer_lat=parsed_clat,  # type: ignore[arg-type]
            customer_lon=parsed_clon,  # type: ignore[arg-type]
            order_hour=parsed_hour,  # type: ignore[arg-type]
            day_of_week=parsed_dow,  # type: ignore[arg-type]
            type_of_order=str(type_of_order_val) if type_of_order_val is not None else "Meal",
            festival=str(festival_val) if festival_val is not None else "No",
            city=str(city_val) if city_val is not None else "Metropolitian",
            order_id=str(order_id_val) if order_id_val is not None else None,
        )

        validation_errors = req.validate()
        if validation_errors:
            return None, validation_errors

        return req, []


@dataclass
class ETAPredictionResponse:
    """Output payload contract returned by the prediction service."""

    predicted_duration_minutes: float
    model_identity: str = "random_forest_regressor"
    distance_km: float = 0.0
    order_id: Optional[str] = None
    prediction_stage: str = "booking_time"
    model_version: str = "0.1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to clean JSON-serializable dictionary."""
        payload: Dict[str, Any] = {
            "predicted_duration_minutes": round(float(self.predicted_duration_minutes), 2),
            "model_identity": self.model_identity,
            "distance_km": round(float(self.distance_km), 4),
            "prediction_stage": self.prediction_stage,
        }
        if self.order_id is not None:
            payload["order_id"] = self.order_id
        return payload
