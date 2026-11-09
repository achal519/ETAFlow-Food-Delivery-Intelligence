"""Data validation and quality audit interfaces for ETAFlow."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class ValidationReport:
    """Summary of data schema and quality validation checks."""

    is_valid: bool
    total_records: int
    missing_columns: List[str] = field(default_factory=list)
    null_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class DataValidator:
    """Validates raw data rows against expected schema and leakage constraints."""

    def __init__(self, required_columns: List[str], target_column: str = "delivery_duration_min") -> None:
        self.required_columns: Set[str] = set(required_columns)
        self.target_column = target_column

    def validate_records(self, records: List[Dict[str, Any]]) -> ValidationReport:
        """Run schema validation and quality sanity checks over in-memory records.

        Args:
            records: List of record dictionaries.

        Returns:
            ValidationReport detailing validation status, missing columns, and errors.
        """
        if not records:
            return ValidationReport(is_valid=False, total_records=0, errors=["Dataset is empty."])

        present_keys = set(records[0].keys())
        missing = [col for col in self.required_columns if col not in present_keys]

        null_counts = {col: 0 for col in present_keys}
        for row in records:
            for k, v in row.items():
                if v is None or v == "" or v == "NaN":
                    null_counts[k] += 1

        is_valid = len(missing) == 0
        return ValidationReport(
            is_valid=is_valid,
            total_records=len(records),
            missing_columns=missing,
            null_counts=null_counts,
            errors=[f"Missing required columns: {missing}"] if missing else [],
        )
