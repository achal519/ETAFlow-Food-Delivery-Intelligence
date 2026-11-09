"""Baseline models for delivery duration regression.

Provides simple heuristic benchmarks against which ML models must be measured.
"""

from typing import List, Optional


class MeanDeliveryDurationBaseline:
    """Baseline regressor predicting the global mean of the target variable.

    Serves as the empirical baseline floor for regression metrics (MAE, RMSE).
    """

    def __init__(self) -> None:
        self.mean_duration: Optional[float] = None

    def fit(self, y: List[float]) -> "MeanDeliveryDurationBaseline":
        """Fit baseline by computing target mean from training labels.

        Args:
            y: List of observed delivery durations in minutes.

        Returns:
            Fitted baseline model.

        Raises:
            ValueError: If target list is empty.
        """
        if not y:
            raise ValueError("Cannot fit baseline on empty target list.")
        self.mean_duration = float(sum(y) / len(y))
        return self

    def predict(self, n_samples: int) -> List[float]:
        """Generate constant baseline predictions.

        Args:
            n_samples: Number of predictions to generate.

        Returns:
            List of constant predicted delivery durations.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self.mean_duration is None:
            raise RuntimeError("Model must be fitted before calling predict.")
        return [self.mean_duration] * n_samples
