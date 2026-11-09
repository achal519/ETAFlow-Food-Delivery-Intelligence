"""Unit tests verifying core ETAFlow module functionality."""

import unittest
from etaflow import __version__
from etaflow.api.schema import ETAPredictionRequest
from etaflow.data.validation import DataValidator
from etaflow.evaluation.metrics import calculate_mae, calculate_rmse, calculate_mape
from etaflow.features.pipeline import calculate_haversine_distance, FeaturePipeline
from etaflow.models.baseline import MeanDeliveryDurationBaseline


class TestSmoke(unittest.TestCase):
    """Smoke tests for basic component readiness."""

    def test_version_present(self) -> None:
        """Verify package version is defined."""
        self.assertEqual(__version__, "0.1.0")

    def test_haversine_calculation(self) -> None:
        """Verify haversine distance returns correct zero and non-zero distances."""
        # Distance between identical points is 0
        dist_zero = calculate_haversine_distance(12.9716, 77.5946, 12.9716, 77.5946)
        self.assertAlmostEqual(dist_zero, 0.0, places=4)

        # Distance between Bangalore and Mysore (~128 km)
        dist = calculate_haversine_distance(12.9716, 77.5946, 12.2958, 76.6394)
        self.assertTrue(120.0 < dist < 150.0)

    def test_baseline_regressor(self) -> None:
        """Verify MeanDeliveryDurationBaseline computes training mean and predicts."""
        model = MeanDeliveryDurationBaseline()
        y_train = [20.0, 30.0, 40.0]
        model.fit(y_train)
        predictions = model.predict(n_samples=2)
        self.assertEqual(len(predictions), 2)
        self.assertAlmostEqual(predictions[0], 30.0)
        self.assertAlmostEqual(predictions[1], 30.0)

    def test_regression_metrics(self) -> None:
        """Verify standard regression metric computations."""
        y_true = [30.0, 40.0]
        y_pred = [35.0, 35.0]
        mae = calculate_mae(y_true, y_pred)
        rmse = calculate_rmse(y_true, y_pred)
        mape = calculate_mape(y_true, y_pred)

        self.assertAlmostEqual(mae, 5.0)
        self.assertAlmostEqual(rmse, 5.0)
        self.assertTrue(mape > 0.0)

    def test_data_validator_schema(self) -> None:
        """Verify validator identifies missing required columns."""
        validator = DataValidator(required_columns=["order_id", "delivery_duration_min"])
        records = [{"order_id": "ORD001"}]
        report = validator.validate_records(records)
        self.assertFalse(report.is_valid)
        self.assertIn("delivery_duration_min", report.missing_columns)

    def test_api_request_validation(self) -> None:
        """Verify ETAPredictionRequest flags out-of-range coordinates."""
        invalid_req = ETAPredictionRequest(
            order_id="TEST01",
            restaurant_lat=95.0, # Invalid latitude > 90
            restaurant_lon=77.0,
            customer_lat=12.0,
            customer_lon=77.0,
            order_hour=14,
            day_of_week=2,
        )
        errors = invalid_req.validate()
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("restaurant_lat" in e for e in errors))

    def test_leakage_columns_excluded(self) -> None:
        """Verify FeaturePipeline strictly excludes delivery_speed, target, and post-booking fields."""
        raw_record = {
            "ID": "0x123",
            "Restaurant_latitude": 12.97,
            "Restaurant_longitude": 77.59,
            "Delivery_location_latitude": 12.98,
            "Delivery_location_longitude": 77.60,
            "Time_taken (min)": 35.0,
            "delivery_speed": "Average",
            "Time_Order_picked": "22:10",
            "Weather_conditions": "Fog",
        }
        pipeline = FeaturePipeline()
        features = pipeline.fit_transform([raw_record])[0]

        self.assertNotIn("delivery_speed", features)
        self.assertNotIn("Time_taken (min)", features)
        self.assertNotIn("Time_Order_picked", features)
        self.assertNotIn("ID", features)
        self.assertIn("Weather_conditions", features)
        self.assertIn("distance_km", features)

    def test_target_extraction(self) -> None:
        """Verify target extraction handles raw 'Time_taken (min)' and canonical names."""
        records = [{"Time_taken (min)": "42.0"}]
        targets = FeaturePipeline.extract_target(records)
        self.assertEqual(targets, [42.0])


if __name__ == "__main__":
    unittest.main()
