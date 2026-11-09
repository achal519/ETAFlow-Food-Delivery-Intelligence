"""Unit tests for ETAFlow baseline regressors, metrics, and model reload."""

import unittest
from pathlib import Path
import tempfile
import numpy as np
import joblib

from etaflow.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_r2,
    compute_regression_metrics,
)
from etaflow.models.regressors import (
    build_dummy_regressor,
    build_ridge_regressor,
    build_random_forest_regressor,
)


class TestModels(unittest.TestCase):
    """Test suite for regression models and evaluation metrics."""

    def test_calculate_r2_known_values(self) -> None:
        """Verify R^2 calculation for perfect, mean, and sub-mean predictions."""
        y_true = [10.0, 20.0, 30.0]

        # 1. Perfect prediction -> R^2 = 1.0
        r2_perfect = calculate_r2(y_true, [10.0, 20.0, 30.0])
        self.assertAlmostEqual(r2_perfect, 1.0)

        # 2. Mean prediction on same set -> R^2 = 0.0
        mean_val = 20.0
        r2_mean = calculate_r2(y_true, [mean_val, mean_val, mean_val])
        self.assertAlmostEqual(r2_mean, 0.0)

        # 3. Very poor prediction -> R^2 < 0.0
        r2_poor = calculate_r2(y_true, [100.0, 100.0, 100.0])
        self.assertTrue(r2_poor < 0.0)

    def test_compute_regression_metrics_keys(self) -> None:
        """Verify compute_regression_metrics returns all four expected metric keys."""
        y_true = [20.0, 30.0]
        y_pred = [22.0, 28.0]
        metrics = compute_regression_metrics(y_true, y_pred)

        self.assertIn("mae_minutes", metrics)
        self.assertIn("rmse_minutes", metrics)
        self.assertIn("mape_percent", metrics)
        self.assertIn("r2_score", metrics)
        self.assertAlmostEqual(metrics["mae_minutes"], 2.0)
        self.assertAlmostEqual(metrics["rmse_minutes"], 2.0)

    def test_dummy_regressor_training_and_shape(self) -> None:
        """Verify DummyRegressor fits on training data and predicts constant mean."""
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_train = np.array([20.0, 30.0, 40.0])
        X_test = np.array([[7.0, 8.0], [9.0, 10.0]])

        dummy = build_dummy_regressor(strategy="mean")
        dummy.fit(X_train, y_train)
        preds = dummy.predict(X_test)

        self.assertEqual(len(preds), 2)
        self.assertEqual(preds.shape, (2,))
        self.assertAlmostEqual(preds[0], 30.0)
        self.assertAlmostEqual(preds[1], 30.0)

    def test_ridge_regressor_training_and_shape(self) -> None:
        """Verify Ridge regressor fits on training features and outputs expected shape."""
        X_train = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]])
        y_train = np.array([15.0, 25.0, 35.0, 45.0])
        X_test = np.array([[5.0, 0.0], [6.0, 1.0]])

        ridge = build_ridge_regressor(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        preds = ridge.predict(X_test)

        self.assertEqual(len(preds), 2)
        self.assertEqual(preds.shape, (2,))
        # Predictions should increase with increasing X
        self.assertTrue(preds[1] > preds[0])

    def test_saved_model_reload_and_inference(self) -> None:
        """Verify serialized Ridge model can be reloaded and yields identical predictions."""
        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_train = np.array([10.0, 20.0, 30.0, 40.0])
        X_test = np.array([[5.0], [6.0]])

        ridge = build_ridge_regressor(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        original_preds = ridge.predict(X_test)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_ridge.joblib"
            joblib.dump(ridge, model_path)

            loaded_model = joblib.load(model_path)
            reloaded_preds = loaded_model.predict(X_test)

            np.testing.assert_allclose(original_preds, reloaded_preds, rtol=1e-5)

    def test_random_forest_training_and_shape(self) -> None:
        """Verify RandomForestRegressor fits on training features and outputs expected shape."""
        X_train = np.array([
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
            [5.0, 2.5],
            [6.0, 3.0],
            [7.0, 3.5],
            [8.0, 4.0],
        ])
        y_train = np.array([12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0])
        X_test = np.array([[2.5, 1.25], [6.5, 3.25]])

        rf = build_random_forest_regressor(
            n_estimators=5,
            max_depth=4,
            min_samples_leaf=1,
            n_jobs=1,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)

        self.assertEqual(len(preds), 2)
        self.assertEqual(preds.shape, (2,))
        # Monotonicity check: larger distance input should have larger predicted duration
        self.assertTrue(preds[1] > preds[0])
        self.assertEqual(len(rf.feature_importances_), 2)

    def test_random_forest_reload_and_inference(self) -> None:
        """Verify serialized RandomForestRegressor can be reloaded and yields identical predictions."""
        X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y_train = np.array([10.0, 15.0, 25.0, 30.0, 40.0, 45.0])
        X_test = np.array([[2.5], [4.5]])

        rf = build_random_forest_regressor(
            n_estimators=5,
            max_depth=4,
            min_samples_leaf=1,
            n_jobs=1,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        original_preds = rf.predict(X_test)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_rf.joblib"
            joblib.dump(rf, model_path)

            loaded_rf = joblib.load(model_path)
            reloaded_preds = loaded_rf.predict(X_test)

            np.testing.assert_allclose(original_preds, reloaded_preds, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
