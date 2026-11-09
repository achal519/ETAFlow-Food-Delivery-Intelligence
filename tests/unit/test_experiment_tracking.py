"""Unit tests for ETAFlow Experiment Tracking (MLflow)."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from etaflow.mlops.experiment import ExperimentTracker
from etaflow.mlops.validator import compute_file_sha256
from pipelines.train_mlflow import run_experiment_tracking_pipeline


class TestExperimentTracking(unittest.TestCase):
    """Test suite for MLflow experiment lifecycle, logging, and summary export."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tracking_uri = f"sqlite:///{Path(self.temp_dir.name) / 'test_mlflow.db'}"
        self.artifact_location = str((Path(self.temp_dir.name) / "artifacts").resolve().as_uri())
        self.experiment_name = "test-etaflow-experiment"

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_tracker_initialization_and_experiment_creation(self) -> None:
        tracker = ExperimentTracker(
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri,
            artifact_location=self.artifact_location,
        )
        self.assertIsNotNone(tracker.experiment_id)
        self.assertEqual(tracker.experiment_name, self.experiment_name)

    def test_log_run_and_summary_retrieval(self) -> None:
        tracker = ExperimentTracker(
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri,
            artifact_location=self.artifact_location,
        )

        dummy_artifact = Path(self.temp_dir.name) / "test_artifact.txt"
        dummy_artifact.write_text("dummy model content", encoding="utf-8")

        run_id = tracker.log_run(
            run_name="test-run-1",
            parameters={"alpha": "1.0", "solver": "auto"},
            metrics={"mae_minutes": 6.5, "r2_score": 0.25},
            tags={"version": "v0.1.0", "stage": "Candidate"},
            artifact_paths=[dummy_artifact],
        )

        self.assertTrue(bool(run_id))
        summaries = tracker.get_runs_summary()
        self.assertEqual(len(summaries), 1)

        summary = summaries[0]
        self.assertEqual(summary["run_name"], "test-run-1")
        self.assertEqual(summary["status"], "FINISHED")
        self.assertEqual(summary["parameters"]["alpha"], "1.0")
        self.assertEqual(summary["metrics"]["mae_minutes"], 6.5)
        self.assertEqual(summary["version"], "v0.1.0")
        self.assertEqual(summary["stage"], "Candidate")

    def test_export_summary_json(self) -> None:
        tracker = ExperimentTracker(
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri,
            artifact_location=self.artifact_location,
        )
        tracker.log_run(
            run_name="test-json-run",
            parameters={"param_a": 10},
            metrics={"test_mae": 5.43},
            tags={"stage": "Archived"},
        )

        out_path = Path(self.temp_dir.name) / "summary.json"
        saved = tracker.export_summary_json(output_path=str(out_path))
        self.assertTrue(saved.exists())

        with open(saved, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["experiment_name"], self.experiment_name)
        self.assertEqual(data["total_runs"], 1)
        self.assertEqual(data["runs"][0]["run_name"], "test-json-run")

    def test_pipeline_preserves_artifact_integrity(self) -> None:
        """Verify running the tracking pipeline does not modify any original model files."""
        model_paths = [
            Path("models/dummy_regressor.joblib"),
            Path("models/ridge_regressor.joblib"),
            Path("models/random_forest_regressor.joblib"),
            Path("models/preprocessor.joblib"),
        ]
        pre_hashes = {str(p): compute_file_sha256(p) for p in model_paths if p.exists()}

        test_report = Path(self.temp_dir.name) / "test_pipeline_summary.json"
        run_experiment_tracking_pipeline(
            tracking_uri=self.tracking_uri,
            summary_report_path=str(test_report),
        )

        self.assertTrue(test_report.exists())
        for p in model_paths:
            if p.exists():
                post_hash = compute_file_sha256(p)
                self.assertEqual(
                    pre_hashes[str(p)],
                    post_hash,
                    f"Artifact {p} was modified during pipeline execution!",
                )


if __name__ == "__main__":
    unittest.main()
