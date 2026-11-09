"""Leakage-safe feature extraction and transformation interfaces.

All transformations maintain strict train/validation separation.
Fit methods compute statistics exclusively on training splits.
"""

import math
from typing import Any, Dict, List, Optional


def calculate_haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate the great-circle distance between two geographic coordinates in kilometers.

    Args:
        lat1: Latitude of point 1 (restaurant) in decimal degrees.
        lon1: Longitude of point 1 (restaurant) in decimal degrees.
        lat2: Latitude of point 2 (delivery location) in decimal degrees.
        lon2: Longitude of point 2 (delivery location) in decimal degrees.

    Returns:
        Distance in kilometers.
    """
    earth_radius_km = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_km * c


# Columns strictly excluded to prevent target leakage
TARGET_LEAKAGE_COLUMNS = {
    "delivery_speed",            # Derived from target / distance; direct target leakage
    "Time_taken (min)",          # Raw target column
    "delivery_duration_min",     # Internal canonical target column
}

# Post-booking timing attributes unavailable at order booking time
POST_BOOKING_COLUMNS = {
    "Time_Order_picked",         # Courier pickup time (only known after kitchen dispatch)
    "pickup_time",
    "actual_pickup_time",
    "actual_transit_time",
    "rider_dropoff_notes",
    "delivery_status_timestamp",
}

# Non-predictive unique identifiers that cause overfitting
IDENTIFIER_COLUMNS = {
    "ID",
    "order_id",
}

# Combined set of disallowed columns for booking-time feature representation
DISALLOWED_FEATURE_COLUMNS = (
    TARGET_LEAKAGE_COLUMNS | POST_BOOKING_COLUMNS | IDENTIFIER_COLUMNS
)


class FeaturePipeline:
    """Extracts features available strictly at prediction time.

    Enforces train/test separation so statistics (such as feature means)
    are fitted exclusively on training records.
    Strictly excludes target columns, target-derived columns (like delivery_speed),
    post-booking timing attributes, and arbitrary record IDs.
    """

    def __init__(self, stage: str = "booking_time") -> None:
        self.stage = stage
        self.fitted_means: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, records: List[Dict[str, Any]]) -> "FeaturePipeline":
        """Fit feature statistics exclusively on training records.

        Args:
            records: Training records list.

        Returns:
            Fitted instance of FeaturePipeline.
        """
        if not records:
            self.is_fitted = True
            return self

        # Calculate baseline means for numerical imputation without data leakage
        distance_vals: List[float] = []
        for r in records:
            if "distance_km" in r and isinstance(r["distance_km"], (int, float)):
                distance_vals.append(float(r["distance_km"]))

        if distance_vals:
            self.fitted_means["distance_km"] = sum(distance_vals) / len(distance_vals)
        else:
            self.fitted_means["distance_km"] = 0.0

        self.is_fitted = True
        return self

    def transform(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform records using fitted statistics without target leakage.

        Strictly filters out target columns, target-derived attributes (e.g. delivery_speed),
        post-booking columns (e.g. Time_Order_picked), and arbitrary identifiers.

        Args:
            records: Input records list.

        Returns:
            Transformed feature dictionaries containing only valid prediction-time features.
        """
        transformed: List[Dict[str, Any]] = []
        for r in records:
            # Build feature row by copying only legitimate prediction-time features
            feature_row: Dict[str, Any] = {}
            for k, v in r.items():
                if k not in DISALLOWED_FEATURE_COLUMNS:
                    feature_row[k] = v

            # Resolve coordinates and compute spatial distance
            res_lat = r.get("restaurant_lat") or r.get("Restaurant_latitude")
            res_lon = r.get("restaurant_lon") or r.get("Restaurant_longitude")
            cust_lat = r.get("customer_lat") or r.get("Delivery_location_latitude")
            cust_lon = r.get("customer_lon") or r.get("Delivery_location_longitude")

            if (
                res_lat is not None
                and res_lon is not None
                and cust_lat is not None
                and cust_lon is not None
            ):
                try:
                    feature_row["distance_km"] = calculate_haversine_distance(
                        float(res_lat),
                        float(res_lon),
                        float(cust_lat),
                        float(cust_lon),
                    )
                except (ValueError, TypeError):
                    feature_row["distance_km"] = self.fitted_means.get("distance_km", 0.0)
            elif "distance_km" not in feature_row:
                feature_row["distance_km"] = self.fitted_means.get("distance_km", 0.0)

            transformed.append(feature_row)
        return transformed

    def fit_transform(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fit on training data and return transformed feature matrix."""
        return self.fit(records).transform(records)

    @staticmethod
    def extract_target(
        records: List[Dict[str, Any]],
        target_column: str = "delivery_duration_min",
    ) -> List[float]:
        """Extract ground-truth target values from records for model training.

        Args:
            records: List of input records containing target labels.
            target_column: Preferred canonical target column name.

        Returns:
            List of float target durations in minutes.

        Raises:
            KeyError: If neither the target_column nor raw 'Time_taken (min)' is present.
        """
        targets: List[float] = []
        for r in records:
            if target_column in r:
                targets.append(float(r[target_column]))
            elif "Time_taken (min)" in r:
                targets.append(float(r["Time_taken (min)"]))
            elif "delivery_duration_min" in r:
                targets.append(float(r["delivery_duration_min"]))
            else:
                raise KeyError(
                    f"Target column '{target_column}' not found in record. "
                    f"Available keys: {list(r.keys())}"
                )
        return targets
