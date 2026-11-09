"""Reproducible MLflow Experiment Tracking Pipeline for ETAFlow.

What this script demonstrates for students:
------------------------------------------
In real-world machine learning teams, we never just train a model and overwrite
old files without tracking what happened. We use MLflow to record:
  - Which algorithms were evaluated (Dummy baseline, Ridge linear regression, Random Forest).
  - What exact settings (hyperparameters) were passed to each algorithm.
  - What exact test metrics (MAE, RMSE, R^2, MAPE) each algorithm achieved on the
    7,793 unseen test rows.
  - The dataset fingerprint (SHA-256) used for evaluation to guarantee data provenance.
  - The saved model artifacts, stored safely inside the MLflow experiment storage (`mlruns/`).

Safety Guarantee:
-----------------
This pipeline reads existing model artifacts, verifies their evaluation scores on
the untouched 7,793 test split, and logs runs to MLflow without overwriting any
production artifacts in `models/` or altering `data/raw/zomato_cleaned.csv`.
"""

from pathlib import Path
import sys
import joblib
import pandas as pd

from etaflow.evaluation.metrics import compute_regression_metrics
from etaflow.mlops.experiment import ExperimentTracker
from etaflow.mlops.validator import compute_file_sha256


def run_experiment_tracking_pipeline(
    raw_csv: str = "data/raw/zomato_cleaned.csv",
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    experiment_name: str = "etaflow-delivery-intelligence",
    tracking_uri: str = "sqlite:///mlflow.db",
    summary_report_path: str = "reports/experiments/experiment_summary.json",
) -> None:
    raw_path = Path(raw_csv)
    proc_path = Path(processed_dir)
    mod_path = Path(models_dir)

    print("=" * 75)
    print("ETAFlow — MLflow Experiment Tracking Pipeline (Phase 2)")
    print("=" * 75)

    # 1. Dataset Fingerprint & Integrity Check
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")
    raw_sha256 = compute_file_sha256(raw_path)
    print(f"[Data Provenance] Raw Dataset Fingerprint (SHA-256): {raw_sha256[:16]}...")

    # 2. Record Pre-Run Artifact Hashes to Guarantee Zero Overwrite
    monitored_artifacts = [
        mod_path / "dummy_regressor.joblib",
        mod_path / "ridge_regressor.joblib",
        mod_path / "random_forest_regressor.joblib",
        mod_path / "preprocessor.joblib",
    ]
    initial_hashes = {}
    for art in monitored_artifacts:
        if not art.exists():
            raise FileNotFoundError(f"Required model artifact not found: {art}")
        initial_hashes[str(art)] = compute_file_sha256(art)

    print("[Integrity Guard] Pre-run hashes verified. All model files are protected.")

    # 3. Load Untouched Test Set (7,793 rows)
    test_feat_path = proc_path / "test_features.csv"
    test_targ_path = proc_path / "test_target.csv"

    if not test_feat_path.exists() or not test_targ_path.exists():
        raise FileNotFoundError("Processed test feature or target files not found!")

    X_test = pd.read_csv(test_feat_path)
    y_test = pd.read_csv(test_targ_path)["delivery_duration_min"].values
    n_test_rows = len(X_test)
    print(f"[Evaluation Split] Loaded untouched test set: {n_test_rows:,} rows, {X_test.shape[1]} features")

    # 4. Initialize MLflow Experiment Tracker
    resolved_uri = (
        tracking_uri
        if tracking_uri.startswith("sqlite:")
        else str(Path(tracking_uri).resolve().as_uri())
    )
    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri=resolved_uri,
    )
    print(f"[MLflow] Active Experiment: '{experiment_name}' (ID: {tracker.experiment_id})")
    print(f"[MLflow] Tracking URI: {tracker.tracking_uri}")

    # 5. Define Model Evaluation Configurations
    experiments_to_log = [
        {
            "run_name": "baseline-dummy",
            "model_path": mod_path / "dummy_regressor.joblib",
            "model_type": "DummyRegressor",
            "version": "v0.1.0",
            "stage": "Archived",
            "parameters": {
                "strategy": "mean",
                "model_class": "DummyRegressor",
            },
            "extra_artifacts": [],
        },
        {
            "run_name": "linear-ridge",
            "model_path": mod_path / "ridge_regressor.joblib",
            "model_type": "Ridge",
            "version": "v0.2.0",
            "stage": "Archived",
            "parameters": {
                "alpha": "1.0",
                "random_state": "42",
                "model_class": "Ridge",
            },
            "extra_artifacts": [],
        },
        {
            "run_name": "champion-random-forest",
            "model_path": mod_path / "random_forest_regressor.joblib",
            "model_type": "RandomForestRegressor",
            "version": "v1.0.0",
            "stage": "Production",
            "parameters": {
                "n_estimators": "100",
                "max_depth": "16",
                "min_samples_leaf": "4",
                "n_jobs": "-1",
                "random_state": "42",
                "model_class": "RandomForestRegressor",
            },
            "extra_artifacts": [mod_path / "preprocessor.joblib"],
        },
    ]

    print("\nEvaluating and logging models to MLflow:")
    print("-" * 75)

    for cfg in experiments_to_log:
        run_name = cfg["run_name"]
        model_file = cfg["model_path"]

        # Load model and predict on test set
        model = joblib.load(model_file)
        preds = model.predict(X_test)
        metrics = compute_regression_metrics(y_test, preds)

        formatted_metrics = {
            "mae_minutes": round(float(metrics["mae_minutes"]), 4),
            "rmse_minutes": round(float(metrics["rmse_minutes"]), 4),
            "r2_score": round(float(metrics["r2_score"]), 4),
            "mape_percent": round(float(metrics["mape_percent"]), 2),
        }

        tags = {
            "model_type": cfg["model_type"],
            "version": cfg["version"],
            "stage": cfg["stage"],
            "dataset_fingerprint": raw_sha256,
            "test_rows": str(n_test_rows),
            "artifact_sha256": initial_hashes[str(model_file)],
        }

        artifacts_to_log = [model_file] + cfg["extra_artifacts"]

        run_id = tracker.log_run(
            run_name=run_name,
            parameters=cfg["parameters"],
            metrics=formatted_metrics,
            tags=tags,
            artifact_paths=artifacts_to_log,
        )

        print(
            f"Logged Run [{run_name:<22}] -> ID: {run_id[:8]}... | "
            f"MAE: {formatted_metrics['mae_minutes']:.4f} min | "
            f"RMSE: {formatted_metrics['rmse_minutes']:.4f} min | "
            f"R^2: {formatted_metrics['r2_score']:.4f}"
        )

    # 6. Post-Run Verification: Zero Overwrite Integrity Check
    print("-" * 75)
    for art in monitored_artifacts:
        post_hash = compute_file_sha256(art)
        if post_hash != initial_hashes[str(art)]:
            raise RuntimeError(f"CRITICAL ERROR: Artifact {art} was altered during MLflow logging!")
    print("[Integrity Guard PASSED] Confirmed: Zero files modified in models/ directory.")

    # 7. Export Experiment Summary JSON
    summary_path = tracker.export_summary_json(output_path=summary_report_path)
    print(f"\n[Report Generated] Experiment summary exported to: {summary_path}")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment_tracking_pipeline()
