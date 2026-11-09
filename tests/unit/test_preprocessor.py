"""Unit tests for booking-time feature preprocessor, target validation, and pipeline isolation."""

import unittest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import tempfile

from etaflow.features.preprocessor import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TIER1_BOOKING_FEATURES,
    FORBIDDEN_LEAKAGE_COLUMNS,
    UNVERIFIED_QUARANTINED_COLUMNS,
    parse_order_time,
    parse_order_date,
    validate_target_series,
    extract_booking_features_and_target,
    build_booking_preprocessor,
)


class TestPreprocessor(unittest.TestCase):
    """Test suite verifying booking-time preprocessing, leakage safety, and serialization."""

    def test_parse_order_time_formats(self) -> None:
        """Verify parsing of standard strings, Excel decimal day fractions, and nulls."""
        # Standard HH:MM strings
        self.assertEqual(parse_order_time("21:55"), 21.0)
        self.assertEqual(parse_order_time("09:20"), 9.0)
        self.assertEqual(parse_order_time("00:15"), 0.0)

        # Excel decimal day fractions
        # 0.458333333 * 24 = 10.99999 -> hour 11
        self.assertEqual(parse_order_time("0.458333333"), 11.0)
        # 0.791666667 * 24 = 19.00000 -> hour 19
        self.assertEqual(parse_order_time("0.791666667"), 19.0)
        # 0.958333333 * 24 = 22.99999 -> hour 23
        self.assertEqual(parse_order_time("0.958333333"), 23.0)

        # Missing / invalid values return None
        self.assertIsNone(parse_order_time(None))
        self.assertIsNone(parse_order_time(""))
        self.assertIsNone(parse_order_time("NaN"))
        self.assertIsNone(parse_order_time("null"))

    def test_parse_order_date(self) -> None:
        """Verify DD-MM-YYYY date parsing into day_of_week and is_weekend."""
        # 13-02-2022 was a Sunday (dow=6, is_weekend=1)
        dow, is_wknd = parse_order_date("13-02-2022")
        self.assertEqual(dow, 6)
        self.assertEqual(is_wknd, 1)

        # 14-02-2022 was a Monday (dow=0, is_weekend=0)
        dow, is_wknd = parse_order_date("14-02-2022")
        self.assertEqual(dow, 0)
        self.assertEqual(is_wknd, 0)

        # Invalid date format raises ValueError
        with self.assertRaises(ValueError):
            parse_order_date("2022-02-13")

    def test_target_validation_explicit_error(self) -> None:
        """Verify target validator raises explicit error on nulls, zeros, or negatives."""
        # Clean target series
        clean_target = pd.Series([20.0, 35.0, 42.0])
        validated = validate_target_series(clean_target)
        self.assertEqual(len(validated), 3)

        # Missing / NaN target records
        null_target = pd.Series([20.0, np.nan, 30.0])
        with self.assertRaises(ValueError) as ctx:
            validate_target_series(null_target)
        self.assertIn("Found 1 missing or NaN target records", str(ctx.exception))

        # Non-positive / zero target records
        zero_target = pd.Series([20.0, 0.0, 30.0])
        with self.assertRaises(ValueError) as ctx:
            validate_target_series(zero_target)
        self.assertIn("Found 1 records with non-positive values", str(ctx.exception))

        # Negative target records
        neg_target = pd.Series([-5.0, 25.0])
        with self.assertRaises(ValueError) as ctx:
            validate_target_series(neg_target)
        self.assertIn("Found 1 records with non-positive values", str(ctx.exception))

    def test_extract_exact_11_features(self) -> None:
        """Verify extracted feature DataFrame contains exactly the 11 Tier 1 booking features."""
        df_mock = pd.DataFrame({
            "ID": ["0x101", "0x102"],
            "Delivery_person_ID": ["DEL01", "DEL02"],
            "Delivery_person_Age": [30.0, 25.0],
            "Delivery_person_Ratings": [4.5, 4.8],
            "Restaurant_latitude": [12.97, 12.98],
            "Restaurant_longitude": [77.59, 77.60],
            "Delivery_location_latitude": [13.00, 13.01],
            "Delivery_location_longitude": [77.62, 77.63],
            "Order_Date": ["13-02-2022", "14-02-2022"],
            "Time_Orderd": ["21:55", "0.458333333"],
            "Time_Order_picked": ["22:10", "11:15"],
            "Weather_conditions": ["Fog", "Sunny"],
            "Road_traffic_density": ["Jam", "Low"],
            "Vehicle_condition": [2, 1],
            "Type_of_order": ["Meal", "Snack"],
            "Type_of_vehicle": ["motorcycle", "scooter"],
            "multiple_deliveries": [1.0, 0.0],
            "Festival": ["No", "No"],
            "City": ["Metropolitian", "Urban"],
            "Time_taken (min)": [35.0, 22.0],
            "distance_km": [4.5, 4.8],
            "delivery_speed": ["Average", "Fast"],
        })

        X, y = extract_booking_features_and_target(df_mock)

        # Check exact column count and names
        self.assertEqual(len(X.columns), 11)
        self.assertEqual(list(X.columns), TIER1_BOOKING_FEATURES)
        self.assertEqual(len(y), 2)
        self.assertEqual(y.name, "delivery_duration_min")

        # Verify forbidden leakage columns are absent
        for forbidden in FORBIDDEN_LEAKAGE_COLUMNS:
            self.assertNotIn(forbidden, X.columns)

        # Verify quarantined columns (weather, traffic) are absent
        for quarantined in UNVERIFIED_QUARANTINED_COLUMNS:
            self.assertNotIn(quarantined, X.columns)

    def test_train_test_isolation(self) -> None:
        """Verify learned imputers and scalers are fitted only on X_train without test leakage."""
        # Train data has order_hour values [10, 20] -> median is 15.0
        X_train = pd.DataFrame({
            "distance_km": [5.0, 15.0],
            "Restaurant_latitude": [12.0, 13.0],
            "Restaurant_longitude": [77.0, 78.0],
            "Delivery_location_latitude": [12.1, 13.1],
            "Delivery_location_longitude": [77.1, 78.1],
            "order_hour": [10.0, 20.0],
            "day_of_week": [1, 2],
            "is_weekend": [0, 0],
            "Type_of_order": ["Meal", "Snack"],
            "Festival": ["No", "No"],
            "City": ["Urban", "Metropolitian"],
        })

        # Test data has a MISSING order_hour (NaN)
        X_test = pd.DataFrame({
            "distance_km": [10.0],
            "Restaurant_latitude": [12.5],
            "Restaurant_longitude": [77.5],
            "Delivery_location_latitude": [12.6],
            "Delivery_location_longitude": [77.6],
            "order_hour": [np.nan],
            "day_of_week": [3],
            "is_weekend": [0],
            "Type_of_order": ["Meal"],
            "Festival": ["No"],
            "City": ["Urban"],
        })

        preprocessor = build_booking_preprocessor()
        preprocessor.fit(X_train)

        # Inspect the fitted median in the numeric pipeline's imputer
        num_transformer = preprocessor.named_transformers_["num"]
        imputer = num_transformer.named_steps["imputer"]
        order_hour_idx = NUMERIC_FEATURES.index("order_hour")
        learned_median_hour = imputer.statistics_[order_hour_idx]

        self.assertEqual(learned_median_hour, 15.0)

        # Transform X_test without fitting
        X_test_trans = preprocessor.transform(X_test)
        self.assertFalse(X_test_trans.isna().any().any())

    def test_preprocessor_reload_and_inference(self) -> None:
        """Verify preprocessor can be serialized to disk, reloaded, and produces identical outputs."""
        X_sample = pd.DataFrame({
            "distance_km": [8.5],
            "Restaurant_latitude": [12.97],
            "Restaurant_longitude": [77.59],
            "Delivery_location_latitude": [13.01],
            "Delivery_location_longitude": [77.63],
            "order_hour": [14.0],
            "day_of_week": [4],
            "is_weekend": [0],
            "Type_of_order": ["Buffet"],
            "Festival": ["No"],
            "City": ["Metropolitian"],
        })

        preprocessor = build_booking_preprocessor()
        preprocessor.fit(X_sample)
        original_output = preprocessor.transform(X_sample)

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir) / "test_preprocessor.joblib"
            joblib.dump(preprocessor, temp_path)

            loaded_preprocessor = joblib.load(temp_path)
            reloaded_output = loaded_preprocessor.transform(X_sample)

            pd.testing.assert_frame_equal(original_output, reloaded_output)


if __name__ == "__main__":
    unittest.main()
