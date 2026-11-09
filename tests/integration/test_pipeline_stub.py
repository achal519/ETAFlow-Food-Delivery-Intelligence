"""Integration test verifying component handoffs from features to baseline and evaluation."""

import unittest
from etaflow.features.pipeline import FeaturePipeline
from etaflow.models.baseline import MeanDeliveryDurationBaseline
from etaflow.evaluation.metrics import compute_regression_metrics


class TestPipelineStub(unittest.TestCase):
    """Integration check across features, baseline modeling, and evaluation."""

    def test_feature_to_baseline_flow(self) -> None:
        """Simulate training handoff: feature transform -> baseline fit -> evaluation."""
        train_records = [
            {
                "order_id": "ORD_01",
                "restaurant_lat": 12.97,
                "restaurant_lon": 77.59,
                "customer_lat": 12.98,
                "customer_lon": 77.60,
                "delivery_duration_min": 25.0,
            },
            {
                "order_id": "ORD_02",
                "restaurant_lat": 12.97,
                "restaurant_lon": 77.59,
                "customer_lat": 13.00,
                "customer_lon": 77.62,
                "delivery_duration_min": 35.0,
            },
        ]

        # 1. Feature pipeline fit and transform
        pipeline = FeaturePipeline()
        transformed = pipeline.fit_transform(train_records)
        self.assertEqual(len(transformed), 2)
        self.assertIn("distance_km", transformed[0])
        self.assertTrue(transformed[0]["distance_km"] > 0)

        # Verify target leakage prevention: target and delivery_speed excluded from features
        self.assertNotIn("delivery_duration_min", transformed[0])
        self.assertNotIn("delivery_speed", transformed[0])

        # 2. Baseline model fit using extracted targets
        y_train = pipeline.extract_target(train_records)
        baseline = MeanDeliveryDurationBaseline()
        baseline.fit(y_train)

        # 3. Predict on evaluation records
        y_pred = baseline.predict(n_samples=len(y_train))
        self.assertEqual(len(y_pred), 2)
        self.assertAlmostEqual(y_pred[0], 30.0)

        # 4. Compute metrics
        metrics = compute_regression_metrics(y_train, y_pred)
        self.assertIn("mae_minutes", metrics)
        self.assertIn("rmse_minutes", metrics)
        self.assertIn("mape_percent", metrics)
        self.assertAlmostEqual(metrics["mae_minutes"], 5.0)


if __name__ == "__main__":
    unittest.main()
