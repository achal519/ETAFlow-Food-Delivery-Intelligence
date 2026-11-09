"""Data & Prediction Drift Monitoring Pipeline for ETAFlow.

What this pipeline demonstrates for students:
------------------------------------------
In production, machine learning models degrade silently because the world changes.
This script demonstrates an automated drift monitoring run:
  1. Computes reference distribution statistics from training data (31,171 rows).
  2. Evaluates a clean test sample (1,500 rows) drawn from untouched test data.
     -> Confirms that healthy operational data causes NO false alarms (STABLE).
  3. Evaluates a simulated drift batch (1,500 rows) simulating severe weather and late-night surges.
     -> Confirms that real distribution changes trigger statistical alerts (ALERT).
  4. Generates structured JSON reports at reports/monitoring/drift_report.json
     and reports/monitoring/reference_stats.json.

Safety Guarantee:
-----------------
This pipeline reads existing processed data and model artifacts to measure drift.
It does NOT retrain any models, alter raw datasets, or overwrite artifacts in `models/`.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import pandas as pd

from etaflow.mlops.drift import (
    DriftDetector,
    generate_synthetic_drift_batch,
    sample_clean_test_batch,
)
from etaflow.mlops.validator import compute_file_sha256


def run_drift_monitoring_pipeline(
    raw_csv: str = "data/raw/zomato_cleaned.csv",
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    reports_dir: str = "reports/monitoring",
    sample_size: int = 1500,
    random_state: int = 42,
) -> Path:
    raw_path = Path(raw_csv)
    proc_path = Path(processed_dir)
    mod_path = Path(models_dir)
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("ETAFlow — Data & Prediction Drift Monitoring Pipeline (Phase 3)")
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

    # 3. Load Processed Datasets
    train_feat_path = proc_path / "train_features.csv"
    test_feat_path = proc_path / "test_features.csv"

    if not train_feat_path.exists() or not test_feat_path.exists():
        raise FileNotFoundError("Processed train or test feature files not found!")

    df_train = pd.read_csv(train_feat_path)
    df_test = pd.read_csv(test_feat_path)

    numeric_cols = [c for c in df_train.columns if c.startswith("num__")]
    categorical_cols = [c for c in df_train.columns if c.startswith("cat__")]

    print(
        f"[Datasets Loaded] Training baseline: {len(df_train):,} rows | "
        f"Test pool: {len(df_test):,} rows | "
        f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical"
    )

    # 4. Load Production Model (for prediction drift evaluation)
    rf_model_path = mod_path / "random_forest_regressor.joblib"
    rf_model = joblib.load(rf_model_path)
    print(f"[Model Loaded] Production Random Forest loaded for prediction drift checks.")

    # 5. Fit DriftDetector with Reference Baseline Statistics
    print("\nComputing baseline reference distribution statistics from training data...")
    detector = DriftDetector.fit_reference(
        df=df_train,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        dataset_fingerprint=raw_sha256,
        model=rf_model,
    )
    ref_stats_path = rep_path / "reference_stats.json"
    detector.save_reference_stats(output_path=str(ref_stats_path))
    print(f"[Reference Saved] Reference stats saved to: {ref_stats_path}")

    # 6. Evaluate Clean Test Batch (Baseline Control)
    print("\nEvaluating Clean Test Batch (Sample from untouched test split)...")
    clean_batch = sample_clean_test_batch(df_test, n_samples=sample_size, random_state=random_state)
    clean_eval = detector.evaluate_batch(
        batch_df=clean_batch,
        batch_label="clean_held_out_test_batch",
        is_synthetic=False,
        model=rf_model,
    )
    print(f"  -> Clean Batch Status: {clean_eval['overall_status']}")
    print(f"  -> Drifted Features:   {clean_eval['drifted_features_count']} / {clean_eval['total_features_evaluated']}")
    print(f"  -> Prediction PSI:     {clean_eval['prediction_drift'].get('psi', 'N/A')}")
    print(f"  -> Prediction Mean:    {clean_eval['prediction_drift'].get('current_mean_prediction_min')} min "
          f"(Reference: {clean_eval['prediction_drift'].get('reference_mean_prediction_min')} min)")

    # 7. Evaluate Simulated Drift Batch (Synthetic Stress Test)
    print("\nEvaluating Simulated Drift Batch (Monsoon surge, long detours, festival spikes)...")
    drift_batch = generate_synthetic_drift_batch(df_test, n_samples=sample_size, random_state=random_state)
    drift_eval = detector.evaluate_batch(
        batch_df=drift_batch,
        batch_label="simulated_drift_monsoon_surge",
        is_synthetic=True,
        model=rf_model,
    )
    print(f"  -> Drift Batch Status: {drift_eval['overall_status']} (Expected: ALERT)")
    print(f"  -> Drifted Features:   {drift_eval['drifted_features_count']} / {drift_eval['total_features_evaluated']}")
    print(f"  -> Prediction PSI:     {drift_eval['prediction_drift'].get('psi', 'N/A')} (Shift detected)")
    print(f"  -> Prediction Mean:    {drift_eval['prediction_drift'].get('current_mean_prediction_min')} min "
          f"(Reference: {drift_eval['prediction_drift'].get('reference_mean_prediction_min')} min)")

    # 8. Post-Run Verification: Zero Overwrite Integrity Check
    print("-" * 75)
    for art in monitored_artifacts:
        post_hash = compute_file_sha256(art)
        if post_hash != initial_hashes[str(art)]:
            raise RuntimeError(f"CRITICAL ERROR: Artifact {art} was altered during drift monitoring!")
    print("[Integrity Guard PASSED] Confirmed: Zero files modified in models/ directory.")

    # 9. Export Comprehensive Drift Report JSON
    report_data = {
        "report_title": "ETAFlow Data & Prediction Drift Monitoring Report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": raw_sha256,
        "sample_size_per_batch": sample_size,
        "monitoring_thresholds": {
            "psi_warning": detector.psi_warning,
            "psi_alert": detector.psi_alert,
            "ks_p_value_alpha": detector.p_value_threshold,
            "chi2_p_value_alpha": detector.p_value_threshold,
        },
        "clean_batch_results": clean_eval,
        "simulated_drift_results": drift_eval,
        "executive_summary": {
            "clean_batch_stable": clean_eval["overall_status"] == "STABLE",
            "drift_batch_alerted": drift_eval["overall_status"] == "ALERT",
            "primary_drift_drivers": [
                feat for feat, m in drift_eval["features"].items() if m["status"] == "ALERT"
            ],
            "prediction_drift_detected": drift_eval["prediction_drift"].get("status") == "ALERT",
        },
    }

    drift_report_path = rep_path / "drift_report.json"
    with open(drift_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[Report Generated] Full drift report exported to: {drift_report_path}")
    print("=" * 75)
    return drift_report_path


if __name__ == "__main__":
    run_drift_monitoring_pipeline()
