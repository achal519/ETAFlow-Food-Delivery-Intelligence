"""Reproducible Data Preparation Pipeline for ETAFlow.

Executes:
  1. Load raw data read-only from data/raw/zomato_cleaned.csv
  2. Validate target integrity (zero silent drops)
  3. Extract exact 11 Tier 1 booking-time features (quarantining weather and traffic)
  4. Perform 80/20 train/test split with random_state=42
  5. Fit ColumnTransformer (imputers, scalers, one-hot encoders) ONLY on X_train
  6. Transform X_test without fitting on it
  7. Export 4 CSV artifacts under data/processed/
  8. Save fitted preprocessor to models/preprocessor.joblib
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from etaflow.features.preprocessor import (
    extract_booking_features_and_target,
    build_booking_preprocessor,
    TIER1_BOOKING_FEATURES,
)


def run_data_preparation(
    raw_data_path: str = "data/raw/zomato_cleaned.csv",
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    raw_path = Path(raw_data_path)
    proc_dir = Path(processed_dir)
    mod_dir = Path(models_dir)

    proc_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")

    print(f"Loading raw dataset from {raw_path}...")
    df_raw = pd.read_csv(raw_path)
    total_raw_rows = len(df_raw)
    print(f"Total raw records loaded: {total_raw_rows}")

    # Extract 11 Tier 1 features and validate target
    print("Extracting Tier 1 booking-time features and validating target...")
    X, y = extract_booking_features_and_target(df_raw, target_col="Time_taken (min)")

    assert len(X) == total_raw_rows, "Mismatch between input rows and extracted feature rows."
    assert len(y) == total_raw_rows, "Mismatch between input rows and validated target rows."
    assert list(X.columns) == TIER1_BOOKING_FEATURES, f"Feature columns do not match Tier 1 list: {X.columns.tolist()}"

    print(f"Validated {len(X)} records with {len(X.columns)} Tier 1 features: {list(X.columns)}")

    # 80/20 Train/Test Split
    print(f"Splitting data into train/test with test_size={test_size}, random_state={random_state}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train set size: {len(X_train)} rows ({len(X_train)/total_raw_rows*100:.1f}%)")
    print(f"Test set size:  {len(X_test)} rows ({len(X_test)/total_raw_rows*100:.1f}%)")

    # Build and fit preprocessor ONLY on X_train
    print("Fitting ColumnTransformer strictly on X_train...")
    preprocessor = build_booking_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)

    print("Transforming X_test without fitting on it...")
    X_test_processed = preprocessor.transform(X_test)

    print(f"Transformed train shape: {X_train_processed.shape}")
    print(f"Transformed test shape:  {X_test_processed.shape}")
    print(f"Transformed feature columns ({len(X_train_processed.columns)}): {list(X_train_processed.columns)}")

    # Save CSV artifacts
    train_feat_path = proc_dir / "train_features.csv"
    train_targ_path = proc_dir / "train_target.csv"
    test_feat_path = proc_dir / "test_features.csv"
    test_targ_path = proc_dir / "test_target.csv"

    print(f"Saving train features to {train_feat_path}...")
    X_train_processed.to_csv(train_feat_path, index=False)

    print(f"Saving train target to {train_targ_path}...")
    y_train.to_frame(name="delivery_duration_min").to_csv(train_targ_path, index=False)

    print(f"Saving test features to {test_feat_path}...")
    X_test_processed.to_csv(test_feat_path, index=False)

    print(f"Saving test target to {test_targ_path}...")
    y_test.to_frame(name="delivery_duration_min").to_csv(test_targ_path, index=False)

    # Save fitted preprocessor
    preprocessor_path = mod_dir / "preprocessor.joblib"
    print(f"Saving fitted preprocessor to {preprocessor_path}...")
    joblib.dump(preprocessor, preprocessor_path)

    # Reload test to verify inference readiness
    loaded_preprocessor = joblib.load(preprocessor_path)
    sample_reloaded_transform = loaded_preprocessor.transform(X_test.iloc[:2])
    assert sample_reloaded_transform.shape == (2, len(X_train_processed.columns)), "Reloaded preprocessor transform mismatch."
    print("Pre-processor reload and inference check passed successfully!")
    print("All preprocessing artifacts saved and verified.")


if __name__ == "__main__":
    run_data_preparation()
