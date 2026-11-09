"""Unit tests for delay intelligence calculations, SLA evaluation, and analytics safeguards."""

import unittest

from etaflow.features.pipeline import calculate_haversine_distance
from etaflow.simulation.engine import SimulationEngine
from etaflow.simulation.order import (
    DemoOrder,
    OrderStatus,
    SLA_ON_TIME_THRESHOLD_MINUTES,
)


class TestDelayIntelligenceAndAnalytics(unittest.TestCase):
    """Test suite covering delay intelligence, SLA compliance, and analytics data integrity."""

    def test_haversine_distance_computation(self) -> None:
        """Verify Haversine distance matches physical geodesic distance between known anchors."""
        # Koramangala (12.9352, 77.6245) to Indiranagar (12.9784, 77.6408) is ~5.07 km
        dist = calculate_haversine_distance(12.9352, 77.6245, 12.9784, 77.6408)
        self.assertAlmostEqual(dist, 5.07, delta=0.2)

        # Zero distance when points coincide
        zero_dist = calculate_haversine_distance(12.9352, 77.6245, 12.9352, 77.6245)
        self.assertEqual(zero_dist, 0.0)

    def test_delay_metric_edge_cases(self) -> None:
        """Verify boundary values for SLA on-time threshold."""
        order = DemoOrder(
            order_id="ORD-EDGE",
            restaurant_lat=12.9,
            restaurant_lon=77.6,
            customer_lat=12.95,
            customer_lon=77.65,
            order_hour=14,
            day_of_week=1,
            predicted_duration_minutes=20.0,
        )

        # Within threshold range [-3.0, +3.0] is On-Time
        order.simulated_actual_duration_minutes = 23.0
        m = order.calculate_delay_metrics()
        self.assertEqual(m["sla_classification"], "On-Time")

        order.simulated_actual_duration_minutes = 17.0
        m_neg_boundary = order.calculate_delay_metrics()
        self.assertEqual(m_neg_boundary["sla_classification"], "On-Time")

        # Just past threshold (+3.01 min is Late)
        order.simulated_actual_duration_minutes = 23.01
        m_late = order.calculate_delay_metrics()
        self.assertEqual(m_late["sla_classification"], "Late")

        # Past negative threshold (-3.01 min is Early)
        order.simulated_actual_duration_minutes = 16.99
        m_early = order.calculate_delay_metrics()
        self.assertEqual(m_early["sla_classification"], "Early")

    def test_fleet_analytics_aggregation(self) -> None:
        """Verify fleet summary metrics when mixed completed, active, and cancelled orders exist."""
        engine = SimulationEngine()

        o1 = DemoOrder("O1", 12.9, 77.6, 12.95, 77.65, 12, 1, predicted_duration_minutes=20.0)
        o1.status = OrderStatus.DELIVERED
        o1.simulated_actual_duration_minutes = 25.0  # +5 min -> Late

        o2 = DemoOrder("O2", 12.9, 77.6, 12.95, 77.65, 12, 1, predicted_duration_minutes=20.0)
        o2.status = OrderStatus.DELIVERED
        o2.simulated_actual_duration_minutes = 21.0  # +1 min -> On-Time

        o3 = DemoOrder("O3", 12.9, 77.6, 12.95, 77.65, 12, 1, predicted_duration_minutes=15.0)
        o3.status = OrderStatus.OUT_FOR_DELIVERY

        o4 = DemoOrder("O4", 12.9, 77.6, 12.95, 77.65, 12, 1, predicted_duration_minutes=18.0)
        o4.status = OrderStatus.CANCELLED

        engine.add_order(o1)
        engine.add_order(o2)
        engine.add_order(o3)
        engine.add_order(o4)

        summary = engine.get_fleet_summary()
        self.assertEqual(summary["total_orders"], 4)
        self.assertEqual(summary["in_transit"], 1)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["cancelled"], 1)
        self.assertEqual(summary["late_orders"], 1)
        self.assertEqual(summary["on_time_rate_pct"], 50.0)
        # Avg predicted: (20 + 20 + 15 + 18) / 4 = 18.25
        self.assertEqual(summary["avg_predicted_duration"], 18.25)
        # Avg simulated (completed only): (25 + 21) / 2 = 23.0
        self.assertEqual(summary["avg_simulated_duration"], 23.0)
        # Avg abs diff (completed only): (|25-20| + |21-20|) / 2 = (5 + 1) / 2 = 3.0
        self.assertEqual(summary["avg_absolute_difference"], 3.0)


if __name__ == "__main__":
    unittest.main()
