"""Evaluation metrics for delivery duration regression models."""

import math
from typing import Dict, Sequence


def calculate_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculate Mean Absolute Error (MAE) in minutes.

    Args:
        y_true: Ground truth delivery durations.
        y_pred: Predicted delivery durations.

    Returns:
        MAE value in minutes.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must match.")
    if len(y_true) == 0:
        raise ValueError("Cannot calculate MAE for empty sequences.")

    return float(sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true))


def calculate_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculate Root Mean Squared Error (RMSE) in minutes.

    Args:
        y_true: Ground truth delivery durations.
        y_pred: Predicted delivery durations.

    Returns:
        RMSE value in minutes.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must match.")
    if len(y_true) == 0:
        raise ValueError("Cannot calculate RMSE for empty sequences.")

    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
    return float(math.sqrt(mse))


def calculate_mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: Ground truth delivery durations.
        y_pred: Predicted delivery durations.

    Returns:
        MAPE value as a percentage.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must match.")
    if len(y_true) == 0:
        raise ValueError("Cannot calculate MAPE for empty sequences.")

    total_pct = 0.0
    valid_count = 0
    for yt, yp in zip(y_true, y_pred):
        if yt > 0:
            total_pct += abs((yt - yp) / yt)
            valid_count += 1

    if valid_count == 0:
        return 0.0
    return float((total_pct / valid_count) * 100.0)


def calculate_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculate Coefficient of Determination (R^2).

    Formula:
        R^2 = 1 - (SS_res / SS_tot)
        where SS_res = sum((y_true - y_pred)^2)
        and SS_tot = sum((y_true - mean(y_true))^2)

    Args:
        y_true: Ground truth delivery durations.
        y_pred: Predicted delivery durations.

    Returns:
        R^2 score (can be negative if predictions perform worse than test mean).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must match.")
    if len(y_true) == 0:
        raise ValueError("Cannot calculate R^2 for empty sequences.")

    mean_true = sum(y_true) / len(y_true)
    ss_tot = sum((yt - mean_true) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))

    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_regression_metrics(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> Dict[str, float]:
    """Compute standardized suite of regression metrics.

    Preserves backward compatibility while providing R^2 score.

    Args:
        y_true: Ground truth delivery durations.
        y_pred: Predicted delivery durations.

    Returns:
        Dictionary containing:
          - 'mae_minutes': Mean Absolute Error in minutes
          - 'rmse_minutes': Root Mean Squared Error in minutes
          - 'mape_percent': Mean Absolute Percentage Error
          - 'r2_score': Coefficient of Determination R^2
    """
    return {
        "mae_minutes": calculate_mae(y_true, y_pred),
        "rmse_minutes": calculate_rmse(y_true, y_pred),
        "mape_percent": calculate_mape(y_true, y_pred),
        "r2_score": calculate_r2(y_true, y_pred),
    }
