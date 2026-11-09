"""Data loading interfaces for ETAFlow.

Handles reading tabular delivery records with schema verification.
Raw data is kept out of Git version control.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import csv


# Default column mappings from verified raw dataset (zomato_cleaned.csv) to internal canonical names
DEFAULT_COLUMN_MAPPING: Dict[str, str] = {
    "Time_taken (min)": "delivery_duration_min",
    "Restaurant_latitude": "restaurant_lat",
    "Restaurant_longitude": "restaurant_lon",
    "Delivery_location_latitude": "customer_lat",
    "Delivery_location_longitude": "customer_lon",
    "Time_Orderd": "order_time",
    "Time_Order_picked": "pickup_time",
}


class DataLoader:
    """Interface for loading raw and processed tabular datasets."""

    def __init__(
        self,
        base_data_dir: str = "data",
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_data_dir = Path(base_data_dir)
        self.column_mapping = (
            column_mapping if column_mapping is not None else DEFAULT_COLUMN_MAPPING
        )

    def load_raw_csv(
        self,
        filename: str,
        apply_mapping: bool = True,
    ) -> List[Dict[str, Any]]:
        """Load records from a raw CSV file.

        Args:
            filename: Name of the CSV file within data/raw.
            apply_mapping: If True, maps raw column names (like 'Time_taken (min)')
                           to canonical names (like 'delivery_duration_min').

        Returns:
            List of dictionaries representing dataset rows.

        Raises:
            FileNotFoundError: If the specified raw data file does not exist.
        """
        file_path = self.base_data_dir / "raw" / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Raw dataset file not found at: {file_path}. "
                "Ensure verified public dataset is placed in data/raw/ before ingestion."
            )

        records: List[Dict[str, Any]] = []
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record: Dict[str, Any] = {}
                for key, val in row.items():
                    target_key = self.column_mapping.get(key, key) if apply_mapping else key

                    # Convert target column to numeric float if possible
                    if target_key == "delivery_duration_min":
                        try:
                            record[target_key] = float(val)
                        except (ValueError, TypeError):
                            record[target_key] = val
                    else:
                        record[target_key] = val
                records.append(record)
        return records
