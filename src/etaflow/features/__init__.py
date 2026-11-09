"""Feature engineering and transformation pipeline for ETAFlow."""

from etaflow.features.pipeline import FeaturePipeline, calculate_haversine_distance
from etaflow.features.preprocessor import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TIER1_BOOKING_FEATURES,
    FORBIDDEN_LEAKAGE_COLUMNS,
    UNVERIFIED_QUARANTINED_COLUMNS,
    parse_order_time,
    parse_order_date,
    compute_haversine_distance,
    validate_target_series,
    extract_booking_features_and_target,
    build_booking_preprocessor,
)

__all__ = [
    "FeaturePipeline",
    "calculate_haversine_distance",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "TIER1_BOOKING_FEATURES",
    "FORBIDDEN_LEAKAGE_COLUMNS",
    "UNVERIFIED_QUARANTINED_COLUMNS",
    "parse_order_time",
    "parse_order_date",
    "compute_haversine_distance",
    "validate_target_series",
    "extract_booking_features_and_target",
    "build_booking_preprocessor",
]
