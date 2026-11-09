"""Leakage-safe booking-time feature extraction and preprocessing for ETAFlow.

Enforces strict prediction-time feature availability, excluding:
  - delivery_speed (target leakage)
  - Time_taken (min) / target columns from inputs
  - Time_Order_picked (post-booking timestamp)
  - Weather_conditions and Road_traffic_density (quarantined as unverified at booking time)
  - Delivery_person_ID, age, ratings, vehicle condition/type (courier unassigned at checkout)
  - ID (arbitrary unique identifier)
"""

from datetime import datetime
from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Exact 11 Tier 1 booking-time modeling features
NUMERIC_FEATURES: List[str] = [
    "distance_km",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "order_hour",
    "day_of_week",
    "is_weekend",
]

CATEGORICAL_FEATURES: List[str] = [
    "Type_of_order",
    "Festival",
    "City",
]

TIER1_BOOKING_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Forbidden columns that must NEVER enter booking-time feature inputs
FORBIDDEN_LEAKAGE_COLUMNS: List[str] = [
    "delivery_speed",
    "Time_taken (min)",
    "delivery_duration_min",
    "Time_Order_picked",
    "ID",
    "order_id",
    "Delivery_person_ID",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "Type_of_vehicle",
    "multiple_deliveries",
]

# Quarantined columns unverified at checkout
UNVERIFIED_QUARANTINED_COLUMNS: List[str] = [
    "Weather_conditions",
    "Road_traffic_density",
]


def parse_order_time(val: Any) -> Optional[float]:
    """Parse raw order time into hour of day (0.0 to 23.0).

    Supports:
      - None, NaN, empty strings -> returns None (imputed on training split)
      - Decimal-day float (e.g., 0.458333333 -> hour 11)
      - HH:MM or HH:MM:SS strings (e.g., '21:55' -> hour 21)
    """
    if val is None:
        return None
    val_str = str(val).strip()
    if val_str == "" or val_str.lower() in {"nan", "null", "none"}:
        return None

    # Check decimal day fraction (Excel serialization artifact)
    try:
        fval = float(val_str)
        if 0.0 <= fval <= 1.0:
            return float(int(round(fval * 24.0)) % 24)
    except ValueError:
        pass

    # Check standard HH:MM or HH:MM:SS
    if ":" in val_str:
        parts = val_str.split(":")
        try:
            hr = int(parts[0])
            if 0 <= hr <= 23:
                return float(hr)
        except ValueError:
            pass

    return None


def parse_order_date(val: Any) -> Tuple[int, int]:
    """Parse DD-MM-YYYY date string into (day_of_week, is_weekend).

    Returns:
        day_of_week: 0 (Monday) to 6 (Sunday)
        is_weekend: 1 if day_of_week in [5, 6] (Saturday, Sunday), else 0

    Raises:
        ValueError: If val is missing or does not match DD-MM-YYYY format.
    """
    if val is None:
        raise ValueError("Order_Date cannot be null.")
    val_str = str(val).strip()
    dt = datetime.strptime(val_str, "%d-%m-%Y")
    dow = dt.weekday()
    is_wknd = 1 if dow in (5, 6) else 0
    return dow, is_wknd


def compute_haversine_distance(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Vectorized calculation of great-circle distance between coordinates in km."""
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def validate_target_series(y_series: pd.Series) -> pd.Series:
    """Validate target values strictly.

    Requires target to be numeric and strictly positive (> 0).
    Raises ValueError with exact count and reason if any row is invalid.
    Never silently drops records.
    """
    if y_series.empty:
        raise ValueError("Target series is empty.")

    null_mask = y_series.isna()
    null_count = int(null_mask.sum())
    if null_count > 0:
        raise ValueError(
            f"Target validation failed: Found {null_count} missing or NaN target records. "
            "Target records cannot be silently dropped."
        )

    try:
        y_numeric = y_series.astype(float)
    except Exception as e:
        raise ValueError(f"Target validation failed: cannot parse all targets as float: {e}")

    non_positive_mask = y_numeric <= 0
    non_positive_count = int(non_positive_mask.sum())
    if non_positive_count > 0:
        raise ValueError(
            f"Target validation failed: Found {non_positive_count} records with non-positive values (<= 0). "
            "Target duration in minutes must be strictly positive."
        )

    return y_numeric


def extract_booking_features_and_target(
    df: pd.DataFrame,
    target_col: str = "Time_taken (min)",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract strictly the 11 Tier 1 booking-time features and validated target.

    Args:
        df: Raw DataFrame containing delivery records.
        target_col: Target column name.

    Returns:
        Tuple of (X, y) where X has exactly 11 columns and y has validated durations.

    Raises:
        KeyError: If target_col or required raw features are missing.
        ValueError: If target values are invalid or non-positive.
    """
    if target_col not in df.columns:
        if "delivery_duration_min" in df.columns:
            target_col = "delivery_duration_min"
        else:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # 1. Validate target explicitly (zero silent drops)
    y = validate_target_series(df[target_col])
    y.name = "delivery_duration_min"

    # 2. Parse temporal features
    order_hours = df["Time_Orderd"].apply(parse_order_time)

    dates_parsed = df["Order_Date"].apply(parse_order_date)
    days_of_week = dates_parsed.apply(lambda x: x[0])
    is_weekends = dates_parsed.apply(lambda x: x[1])

    # 3. Compute distance from coordinates
    computed_dist = compute_haversine_distance(
        df["Restaurant_latitude"].astype(float),
        df["Restaurant_longitude"].astype(float),
        df["Delivery_location_latitude"].astype(float),
        df["Delivery_location_longitude"].astype(float),
    )

    # 4. Construct exact 11 Tier 1 booking-time feature DataFrame
    X = pd.DataFrame({
        "distance_km": computed_dist,
        "Restaurant_latitude": df["Restaurant_latitude"].astype(float),
        "Restaurant_longitude": df["Restaurant_longitude"].astype(float),
        "Delivery_location_latitude": df["Delivery_location_latitude"].astype(float),
        "Delivery_location_longitude": df["Delivery_location_longitude"].astype(float),
        "order_hour": order_hours,
        "day_of_week": days_of_week.astype(int),
        "is_weekend": is_weekends.astype(int),
        "Type_of_order": df["Type_of_order"].astype(str).str.strip(),
        "Festival": df["Festival"].astype(str).str.strip(),
        "City": df["City"].astype(str).str.strip(),
    })

    return X, y


def build_booking_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that imputes and scales numeric features and one-hot encodes categoricals.

    Numeric pipeline:
      - SimpleImputer(strategy='median')
      - StandardScaler()

    Categorical pipeline:
      - SimpleImputer(strategy='most_frequent')
      - OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    Returns:
        ColumnTransformer configured with transform output as pandas DataFrame.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor
