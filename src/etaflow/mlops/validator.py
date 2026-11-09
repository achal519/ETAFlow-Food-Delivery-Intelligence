"""Data validation, schema enforcement, and dataset SHA-256 fingerprinting for ETAFlow."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd


ALLOWED_ORDER_TYPES: Set[str] = {"Meal", "Snack", "Drinks", "Buffet"}
ALLOWED_FESTIVAL_STATUSES: Set[str] = {"No", "Yes"}
ALLOWED_CITIES: Set[str] = {"Metropolitian", "Urban", "Semi-Urban"}

# Expected raw schema columns in data/raw/zomato_cleaned.csv
EXPECTED_RAW_COLUMNS: List[str] = [
    "ID",
    "Delivery_person_ID",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
    "Weather_conditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
    "Time_taken (min)",
    "distance_km",
    "delivery_speed",
]

# Strict Tier 1 columns required for booking-time inference pipeline
REQUIRED_TIER1_COLUMNS: List[str] = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Type_of_order",
    "Festival",
    "City",
    "Time_taken (min)",
]


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA-256 hexadecimal hash of a file efficiently in binary chunks.

    Args:
        file_path: Path to the target file.
        chunk_size: Byte size per chunk (default: 64 KB).

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for SHA-256 computation: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class ValidationReport:
    """Structured report detailing dataset schema, ranges, missingness, and SHA-256 fingerprint."""

    is_valid: bool
    dataset_path: str
    dataset_sha256: str
    total_rows: int
    total_columns: int
    validation_timestamp_utc: str
    present_columns: List[str] = field(default_factory=list)
    missing_required_columns: List[str] = field(default_factory=list)
    null_counts: Dict[str, int] = field(default_factory=dict)
    schema_errors: List[str] = field(default_factory=list)
    range_errors: List[str] = field(default_factory=list)
    category_errors: List[str] = field(default_factory=list)
    summary_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert report dataclass to a JSON-serializable dictionary."""
        return asdict(self)

    def save_json(self, destination_path: Union[str, Path]) -> Path:
        """Serialize validation report to JSON on disk."""
        dest = Path(destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return dest


class DatasetValidator:
    """Validates raw datasets against schema constraints, bounds, allowed categories, and target sanity."""

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        min_target_value: float = 1.0,
        max_target_value: float = 180.0,
        min_latitude: float = 8.0,
        max_latitude: float = 38.0,
        min_longitude: float = 68.0,
        max_longitude: float = 98.0,
    ) -> None:
        self.required_columns = required_columns or REQUIRED_TIER1_COLUMNS
        self.min_target_value = min_target_value
        self.max_target_value = max_target_value
        self.min_lat = min_latitude
        self.max_lat = max_latitude
        self.min_lon = min_longitude
        self.max_lon = max_longitude

    def validate_file(self, file_path: Union[str, Path]) -> ValidationReport:
        """Perform comprehensive data validation directly on a CSV file.

        Args:
            file_path: Path to the CSV dataset.

        Returns:
            ValidationReport instance.
        """
        path = Path(file_path)
        if not path.is_file():
            return ValidationReport(
                is_valid=False,
                dataset_path=str(path),
                dataset_sha256="",
                total_rows=0,
                total_columns=0,
                validation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                schema_errors=[f"Dataset file does not exist: {path}"],
                summary_message="Validation failed: file not found.",
            )

        file_hash = compute_file_sha256(path)
        df = pd.read_csv(path)
        return self.validate_dataframe(df, dataset_path=str(path), dataset_sha256=file_hash)

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        dataset_path: str = "in_memory",
        dataset_sha256: str = "",
    ) -> ValidationReport:
        """Validate an in-memory DataFrame against schema, value bounds, and categories."""
        schema_errors: List[str] = []
        range_errors: List[str] = []
        category_errors: List[str] = []

        present_cols = list(df.columns)
        missing_required = [col for col in self.required_columns if col not in present_cols]
        if missing_required:
            schema_errors.append(f"Missing required columns: {missing_required}")

        null_counts = df.isnull().sum().to_dict()

        # Check target sanity
        target_col = "Time_taken (min)"
        if target_col in df.columns:
            target_series = pd.to_numeric(df[target_col], errors="coerce")
            target_nulls = int(target_series.isnull().sum())
            if target_nulls > 0:
                schema_errors.append(f"Target column '{target_col}' has {target_nulls} null or non-numeric values.")

            valid_targets = target_series.dropna()
            out_of_bounds = valid_targets[
                (valid_targets < self.min_target_value) | (valid_targets > self.max_target_value)
            ]
            if not out_of_bounds.empty:
                range_errors.append(
                    f"Target values out of valid range [{self.min_target_value}, {self.max_target_value}]: "
                    f"{len(out_of_bounds)} anomalous rows found."
                )

        # Check coordinate bounds
        for lat_col in ["Restaurant_latitude", "Delivery_location_latitude"]:
            if lat_col in df.columns:
                lat_vals = pd.to_numeric(df[lat_col], errors="coerce").dropna()
                bad_lats = lat_vals[(lat_vals < self.min_lat) | (lat_vals > self.max_lat)]
                if not bad_lats.empty:
                    range_errors.append(
                        f"Latitude column '{lat_col}' has {len(bad_lats)} values outside [{self.min_lat}, {self.max_lat}]."
                    )

        for lon_col in ["Restaurant_longitude", "Delivery_location_longitude"]:
            if lon_col in df.columns:
                lon_vals = pd.to_numeric(df[lon_col], errors="coerce").dropna()
                bad_lons = lon_vals[(lon_vals < self.min_lon) | (lon_vals > self.max_lon)]
                if not bad_lons.empty:
                    range_errors.append(
                        f"Longitude column '{lon_col}' has {len(bad_lons)} values outside [{self.min_lon}, {self.max_lon}]."
                    )

        # Check allowed categories
        if "Type_of_order" in df.columns:
            unseen_types = set(df["Type_of_order"].dropna().astype(str).unique()) - ALLOWED_ORDER_TYPES
            if unseen_types:
                category_errors.append(f"Unrecognized values in Type_of_order: {list(unseen_types)}")

        if "Festival" in df.columns:
            unseen_festivals = set(df["Festival"].dropna().astype(str).unique()) - ALLOWED_FESTIVAL_STATUSES
            if unseen_festivals:
                category_errors.append(f"Unrecognized values in Festival: {list(unseen_festivals)}")

        if "City" in df.columns:
            unseen_cities = set(df["City"].dropna().astype(str).unique()) - ALLOWED_CITIES
            if unseen_cities:
                category_errors.append(f"Unrecognized values in City: {list(unseen_cities)}")

        is_valid = len(schema_errors) == 0 and len(range_errors) == 0 and len(category_errors) == 0
        summary = (
            f"Dataset PASSED validation: {len(df)} rows, {len(df.columns)} columns verified."
            if is_valid
            else f"Dataset FAILED validation: {len(schema_errors)} schema, {len(range_errors)} range, {len(category_errors)} category errors."
        )

        return ValidationReport(
            is_valid=is_valid,
            dataset_path=dataset_path,
            dataset_sha256=dataset_sha256,
            total_rows=len(df),
            total_columns=len(df.columns),
            validation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            present_columns=present_cols,
            missing_required_columns=missing_required,
            null_counts=null_counts,
            schema_errors=schema_errors,
            range_errors=range_errors,
            category_errors=category_errors,
            summary_message=summary,
        )
