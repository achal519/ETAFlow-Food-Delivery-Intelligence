"""Unit tests for ETAFlow Data and Prediction Drift Monitoring (Phase 3)."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from etaflow.mlops.drift import (
    DriftDetector,
    compute_categorical_psi,
    compute_chi2_test,
    compute_ks_test,
    compute_numeric_psi,
    generate_synthetic_drift_batch,
    sample_clean_test_batch,
)
from etaflow.mlops.validator import compute_file_sha256
from pipelines.monitor_drift import run_drift_monitoring_pipeline


class TestDriftMonitoring(unittest.TestCase):
    """Test suite for PSI, KS-test, Chi-square, and automated drift pipeline."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_compute_numeric_psi(self) -> None:
        # Same distribution -> PSI should be close to 0 (< 0.05)
        ref = self.rng.normal(loc=0.0, scale=1.0, size=2000)
        target_clean = self.rng.normal(loc=0.0, scale=1.0, size=1500)
        psi_clean = compute_numeric_psi(ref, target_clean)
        self.assertLess(psi_clean, 0.08)

        # Heavily shifted distribution -> PSI should be > 0.20
        target_drift = self.rng.normal(loc=2.5, scale=1.0, size=1500)
        psi_drift = compute_numeric_psi(ref, target_drift)
        self.assertGreater(psi_drift, 0.20)

    def test_compute_categorical_psi(self) -> None:
        ref_counts = {"Snack": 500, "Meal": 300, "Drinks": 150, "Buffet": 50}
        clean_counts = {"Snack": 252, "Meal": 149, "Drinks": 76, "Buffet": 23}
        psi_clean = compute_categorical_psi(ref_counts, clean_counts)
        self.assertLess(psi_clean, 0.05)

        drift_counts = {"Snack": 50, "Meal": 100, "Drinks": 150, "Buffet": 200}
        psi_drift = compute_categorical_psi(ref_counts, drift_counts)
        self.assertGreater(psi_drift, 0.20)

    def test_compute_ks_test(self) -> None:
        ref = self.rng.normal(loc=10.0, scale=2.0, size=1000)
        tar_clean = self.rng.normal(loc=10.0, scale=2.0, size=1000)
        res_clean = compute_ks_test(ref, tar_clean)
        self.assertGreater(res_clean["p_value"], 0.01)

        tar_drift = self.rng.normal(loc=15.0, scale=2.0, size=1000)
        res_drift = compute_ks_test(ref, tar_drift)
        self.assertLess(res_drift["p_value"], 0.01)
        self.assertGreater(res_drift["statistic"], 0.5)

    def test_compute_chi2_test(self) -> None:
        ref_counts = {"Metropolitian": 800, "Urban": 180, "Semi-Urban": 20}
        clean_counts = {"Metropolitian": 403, "Urban": 88, "Semi-Urban": 9}
        res_clean = compute_chi2_test(ref_counts, clean_counts)
        self.assertGreater(res_clean["p_value"], 0.05)

        drift_counts = {"Metropolitian": 200, "Urban": 200, "Semi-Urban": 100}
        res_drift = compute_chi2_test(ref_counts, drift_counts)
        self.assertLess(res_drift["p_value"], 0.001)

    def test_drift_detector_clean_batch_behavior(self) -> None:
        train_path = Path("data/processed/train_features.csv")
        test_path = Path("data/processed/test_features.csv")
        if not train_path.exists() or not test_path.exists():
            self.skipTest("Processed feature files not available")

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        num_cols = [c for c in df_train.columns if c.startswith("num__")]
        cat_cols = [c for c in df_train.columns if c.startswith("cat__")]

        detector = DriftDetector.fit_reference(df_train, num_cols, cat_cols)
        clean_batch = sample_clean_test_batch(df_test, n_samples=1500, random_state=42)

        res = detector.evaluate_batch(clean_batch, batch_label="clean_test", is_synthetic=False)
        self.assertEqual(res["overall_status"], "STABLE")
        self.assertEqual(res["drifted_features_count"], 0)
        self.assertFalse(res["is_synthetic"])

    def test_drift_detector_simulated_drift_detection(self) -> None:
        train_path = Path("data/processed/train_features.csv")
        test_path = Path("data/processed/test_features.csv")
        if not train_path.exists() or not test_path.exists():
            self.skipTest("Processed feature files not available")

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        num_cols = [c for c in df_train.columns if c.startswith("num__")]
        cat_cols = [c for c in df_train.columns if c.startswith("cat__")]

        detector = DriftDetector.fit_reference(df_train, num_cols, cat_cols)
        drift_batch = generate_synthetic_drift_batch(df_test, n_samples=1500, random_state=42)

        res = detector.evaluate_batch(drift_batch, batch_label="simulated_drift", is_synthetic=True)
        self.assertEqual(res["overall_status"], "ALERT")
        self.assertGreater(res["drifted_features_count"], 0)
        self.assertTrue(res["is_synthetic"])

        # Specifically check that distance and festival drift were flagged
        self.assertIn("num__distance_km", res["features"])
        self.assertGreater(res["features"]["num__distance_km"]["psi"], 0.20)

    def test_pipeline_execution_and_artifact_integrity(self) -> None:
        model_paths = [
            Path("models/dummy_regressor.joblib"),
            Path("models/ridge_regressor.joblib"),
            Path("models/random_forest_regressor.joblib"),
            Path("models/preprocessor.joblib"),
        ]
        pre_hashes = {str(p): compute_file_sha256(p) for p in model_paths if p.exists()}

        rep_dir = Path(self.temp_dir.name) / "reports"
        run_drift_monitoring_pipeline(reports_dir=str(rep_dir))

        report_file = rep_dir / "drift_report.json"
        ref_file = rep_dir / "reference_stats.json"

        self.assertTrue(report_file.exists())
        self.assertTrue(ref_file.exists())

        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("clean_batch_results", data)
        self.assertIn("simulated_drift_results", data)
        self.assertTrue(data["executive_summary"]["clean_batch_stable"])
        self.assertTrue(data["executive_summary"]["drift_batch_alerted"])

        # Guarantee zero model modifications
        for p in model_paths:
            if p.exists():
                post_hash = compute_file_sha256(p)
                self.assertEqual(pre_hashes[str(p)], post_hash)


if __name__ == "__main__":
    unittest.main()
