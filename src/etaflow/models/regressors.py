"""Model constructor interfaces for baseline, linear, and ensemble regression in ETAFlow."""

from typing import Optional
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def build_dummy_regressor(strategy: str = "mean") -> DummyRegressor:
    """Build a baseline DummyRegressor that predicts constant target statistics.

    Args:
        strategy: Prediction strategy ('mean', 'median'). Defaults to 'mean'.

    Returns:
        Configured DummyRegressor instance.
    """
    return DummyRegressor(strategy=strategy)


def build_ridge_regressor(
    alpha: float = 1.0,
    random_state: int = 42,
) -> Ridge:
    """Build a regularized Ridge linear regression model.

    Args:
        alpha: L2 regularization strength. Defaults to 1.0.
        random_state: Random state for solver consistency. Defaults to 42.

    Returns:
        Configured Ridge regression instance.
    """
    return Ridge(alpha=alpha, random_state=random_state)


def build_random_forest_regressor(
    n_estimators: int = 100,
    max_depth: Optional[int] = 16,
    min_samples_leaf: int = 4,
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Build a regularized RandomForestRegressor ensemble model.

    Args:
        n_estimators: Number of trees in the forest. Defaults to 100.
        max_depth: Maximum depth of individual trees. Defaults to 16.
        min_samples_leaf: Minimum samples required at each leaf node. Defaults to 4.
        n_jobs: Number of parallel jobs (-1 for all cores). Defaults to -1.
        random_state: Random state for deterministic splits. Defaults to 42.

    Returns:
        Configured RandomForestRegressor instance.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
    )
