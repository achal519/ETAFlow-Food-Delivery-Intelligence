"""Unit and integration tests for ETAFlow Prediction API and Service."""

import json
from pathlib import Path
import threading
from urllib.error import HTTPError
import urllib.request
import unittest
import joblib
import pandas as pd

from etaflow.api.app import create_api_server
from etaflow.api.schema import ETAPredictionRequest, ETAPredictionResponse
from etaflow.api.service import (
    PREPROCESSOR_INPUT_COLUMNS,
    ArtifactUnavailableError,
    ETAPredictionService,
)


class TestETAPredictionAPI(unittest.TestCase):
    """Test suite covering ETAPredictionService, schema contracts, and HTTP endpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start a test server on a free port in a background daemon thread."""
        cls.service = ETAPredictionService()
        cls.server = create_api_server(service=cls.service, host="127.0.0.1", port=0)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"

        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        """Shut down the test server."""
        cls.server.shutdown()
        cls.server.server_close()

    def _http_request(
        self,
        method: str,
        path: str,
        payload: dict = None,
        raw_body: bytes = None,
    ) -> tuple[int, dict]:
        """Helper to send HTTP requests to test server and return (status, json_dict)."""
        url = f"{self.base_url}{path}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif raw_body is not None:
            data = raw_body

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                status = resp.status
                body = resp.read().decode("utf-8")
                return status, json.loads(body) if body else {}
        except HTTPError as exc:
            body = exc.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            return exc.code, parsed

    def _http_request_raw(
        self,
        method: str,
        path: str,
        headers: dict = None,
        payload: dict = None,
        raw_body: bytes = None,
    ) -> tuple[int, dict, dict]:
        """Helper to send HTTP requests and return (status, parsed_json_or_empty, response_headers)."""
        url = f"{self.base_url}{path}"
        data = None
        req_headers = dict(headers or {})
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        elif raw_body is not None:
            data = raw_body

        req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                status = resp.status
                body = resp.read().decode("utf-8")
                resp_headers = {k.title(): v for k, v in resp.headers.items()}
                return status, json.loads(body) if body else {}, resp_headers
        except HTTPError as exc:
            body = exc.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            exc_headers = {k.title(): v for k, v in exc.headers.items()}
            return exc.code, parsed, exc_headers

    # 1. Health checks
    def test_health_check_healthy(self) -> None:
        """GET /health returns 200 with healthy status when artifacts are present."""
        status, data = self._http_request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("artifacts_loaded"))
        self.assertEqual(data.get("model_identity"), "random_forest_regressor")

    def test_health_check_missing_artifacts(self) -> None:
        """Service with missing artifact paths returns 503 on health check and predict."""
        dummy_service = ETAPredictionService(
            preprocessor_path=Path("non_existent_preprocessor.joblib"),
            model_path=Path("non_existent_model.joblib"),
        )
        self.assertFalse(dummy_service.is_ready)

        test_server = create_api_server(service=dummy_service, host="127.0.0.1", port=0)
        port = test_server.server_address[1]
        t = threading.Thread(target=test_server.serve_forever, daemon=True)
        t.start()

        try:
            req_health = urllib.request.Request(f"http://127.0.0.1:{port}/health", method="GET")
            with self.assertRaises(HTTPError) as ctx:
                urllib.request.urlopen(req_health)
            self.assertEqual(ctx.exception.code, 503)

            valid_payload = {
                "restaurant_lat": 12.9716,
                "restaurant_lon": 77.5946,
                "customer_lat": 12.9816,
                "customer_lon": 77.6046,
                "order_hour": 19,
                "day_of_week": 4,
            }
            req_predict = urllib.request.Request(
                f"http://127.0.0.1:{port}/predict",
                data=json.dumps(valid_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as ctx2:
                urllib.request.urlopen(req_predict)
            self.assertEqual(ctx2.exception.code, 503)
        finally:
            test_server.shutdown()
            test_server.server_close()

    # 2. Valid Prediction
    def test_predict_valid_payload(self) -> None:
        """POST /predict returns 200 with expected response fields."""
        payload = {
            "order_id": "ORD_TEST_01",
            "restaurant_lat": 12.9716,
            "restaurant_lon": 77.5946,
            "customer_lat": 12.9816,
            "customer_lon": 77.6046,
            "order_hour": 19,
            "day_of_week": 4,
            "type_of_order": "Meal",
            "festival": "No",
            "city": "Metropolitian",
        }
        status, data = self._http_request("POST", "/predict", payload=payload)
        self.assertEqual(status, 200)
        self.assertIn("predicted_duration_minutes", data)
        self.assertGreater(data["predicted_duration_minutes"], 0.0)
        self.assertEqual(data.get("model_identity"), "random_forest_regressor")
        self.assertEqual(data.get("order_id"), "ORD_TEST_01")
        self.assertEqual(data.get("prediction_stage"), "booking_time")
        self.assertGreater(data.get("distance_km"), 0.0)
        # Verify no uncalibrated interval claims
        self.assertNotIn("confidence_interval", data)
        self.assertNotIn("std_dev", data)

    # 3. Forbidden Post-Booking Fields Rejection
    def test_predict_forbidden_fields(self) -> None:
        """POST /predict with forbidden post-booking fields returns HTTP 400."""
        forbidden_test_cases = [
            {"Time_Order_picked": "19:15"},
            {"delivery_speed": "Fast"},
            {"delivery_duration_min": 25.0},
            {"actual_delivery_duration": 30.0},
            {"Delivery_person_Ratings": 4.8},
            {"courier_id": "RIDER_99"},
            {"Delivery_person_ID": "IND99"},
            {"rider_rating": 4.5},
        ]
        base_valid = {
            "restaurant_lat": 12.9716,
            "restaurant_lon": 77.5946,
            "customer_lat": 12.9816,
            "customer_lon": 77.6046,
            "order_hour": 19,
            "day_of_week": 4,
        }
        for bad_field in forbidden_test_cases:
            with self.subTest(bad_field=bad_field):
                payload = dict(base_valid)
                payload.update(bad_field)
                status, data = self._http_request("POST", "/predict", payload=payload)
                self.assertEqual(status, 400)
                self.assertIn("details", data)
                self.assertTrue(any("Forbidden" in str(d) for d in data["details"]))

    # 4. Missing required fields
    def test_predict_missing_required_fields(self) -> None:
        """POST /predict missing required fields returns HTTP 400 with details."""
        payload = {
            "restaurant_lat": 12.9716,
            # Missing restaurant_lon
            "customer_lat": 12.9816,
            "customer_lon": 77.6046,
            # Missing order_hour
            "day_of_week": 4,
        }
        status, data = self._http_request("POST", "/predict", payload=payload)
        self.assertEqual(status, 400)
        self.assertIn("details", data)
        details_str = " ".join(data["details"])
        self.assertIn("restaurant_lon", details_str)
        self.assertIn("order_hour", details_str)

    # 5. Invalid coordinates and timing bounds
    def test_predict_invalid_bounds(self) -> None:
        """POST /predict with out-of-range coordinates or timing returns HTTP 400."""
        out_of_bounds_cases = [
            ({"restaurant_lat": 95.0}, "restaurant_lat"),
            ({"restaurant_lon": 185.0}, "restaurant_lon"),
            ({"customer_lat": -95.0}, "customer_lat"),
            ({"customer_lon": -185.0}, "customer_lon"),
            ({"order_hour": 24}, "order_hour"),
            ({"day_of_week": 7}, "day_of_week"),
            ({"day_of_week": -1}, "day_of_week"),
        ]
        base = {
            "restaurant_lat": 12.9716,
            "restaurant_lon": 77.5946,
            "customer_lat": 12.9816,
            "customer_lon": 77.6046,
            "order_hour": 19,
            "day_of_week": 4,
        }
        for bad_val, expected_key in out_of_bounds_cases:
            with self.subTest(bad_val=bad_val):
                payload = dict(base)
                payload.update(bad_val)
                status, data = self._http_request("POST", "/predict", payload=payload)
                self.assertEqual(status, 400)
                self.assertTrue(any(expected_key in d for d in data.get("details", [])))

    # 6. Non-finite values
    def test_predict_non_finite_values(self) -> None:
        """POST /predict with NaN or Inf numbers returns HTTP 400."""
        payload_nan = {
            "restaurant_lat": "NaN",
            "restaurant_lon": 77.5946,
            "customer_lat": 12.9816,
            "customer_lon": 77.6046,
            "order_hour": 19,
            "day_of_week": 4,
        }
        status, data = self._http_request("POST", "/predict", payload=payload_nan)
        self.assertEqual(status, 400)

    # 7. Invalid categorical domains
    def test_predict_invalid_categorical_values(self) -> None:
        """POST /predict with unknown category values returns HTTP 400."""
        base = {
            "restaurant_lat": 12.9716,
            "restaurant_lon": 77.5946,
            "customer_lat": 12.9816,
            "customer_lon": 77.6046,
            "order_hour": 19,
            "day_of_week": 4,
        }
        invalid_cat_cases = [
            ({"type_of_order": "Pizza"}, "type_of_order"),
            ({"festival": "Maybe"}, "festival"),
            ({"city": "Gotham"}, "city"),
        ]
        for bad_cat, expected_key in invalid_cat_cases:
            with self.subTest(bad_cat=bad_cat):
                payload = dict(base)
                payload.update(bad_cat)
                status, data = self._http_request("POST", "/predict", payload=payload)
                self.assertEqual(status, 400)
                self.assertTrue(any(expected_key in d for d in data.get("details", [])))

    # 8. Feature column order and derivation
    def test_feature_column_order_and_weekend(self) -> None:
        """Service.prepare_feature_dataframe yields exact 11 columns in exact order."""
        req_weekday = ETAPredictionRequest(
            restaurant_lat=12.9716,
            restaurant_lon=77.5946,
            customer_lat=12.9816,
            customer_lon=77.6046,
            order_hour=14,
            day_of_week=2,  # Wednesday
            type_of_order="Snack",
            festival="No",
            city="Urban",
        )
        df_weekday = self.service.prepare_feature_dataframe(req_weekday)
        self.assertEqual(list(df_weekday.columns), PREPROCESSOR_INPUT_COLUMNS)
        self.assertEqual(df_weekday["is_weekend"].iloc[0], 0)

        req_weekend = ETAPredictionRequest(
            restaurant_lat=12.9716,
            restaurant_lon=77.5946,
            customer_lat=12.9816,
            customer_lon=77.6046,
            order_hour=20,
            day_of_week=5,  # Saturday
            type_of_order="Meal",
            festival="Yes",
            city="Metropolitian",
        )
        df_weekend = self.service.prepare_feature_dataframe(req_weekend)
        self.assertEqual(df_weekend["is_weekend"].iloc[0], 1)

    # 9. Inference consistency with direct model call
    def test_inference_consistency(self) -> None:
        """Prediction service output exactly matches direct preprocessor + model inference."""
        payload = {
            "restaurant_lat": 12.9716,
            "restaurant_lon": 77.5946,
            "customer_lat": 12.9816,
            "customer_lon": 77.6046,
            "order_hour": 18,
            "day_of_week": 3,
            "type_of_order": "Meal",
            "festival": "No",
            "city": "Metropolitian",
        }
        # Direct inference
        req, errors = ETAPredictionRequest.from_dict(payload)
        self.assertEqual(len(errors), 0)
        df_11 = self.service.prepare_feature_dataframe(req)

        direct_prep = joblib.load("models/preprocessor.joblib")
        direct_rf = joblib.load("models/random_forest_regressor.joblib")

        transformed = direct_prep.transform(df_11)
        expected_pred = float(direct_rf.predict(pd.DataFrame(transformed, columns=direct_prep.get_feature_names_out()))[0])

        # Service inference
        service_resp = self.service.predict(req)
        self.assertAlmostEqual(service_resp.predicted_duration_minutes, expected_pred, places=4)

        # HTTP inference
        status, data = self._http_request("POST", "/predict", payload=payload)
        self.assertEqual(status, 200)
        self.assertAlmostEqual(data["predicted_duration_minutes"], round(expected_pred, 2), places=2)

    # 10. Invalid JSON syntax
    def test_invalid_json_payload(self) -> None:
        """POST /predict with malformed JSON returns HTTP 400."""
        status, data = self._http_request("POST", "/predict", raw_body=b"{bad-json: true")
        self.assertEqual(status, 400)
        self.assertIn("error", data)

    # 11. Unknown endpoint
    def test_not_found(self) -> None:
        """Request to unknown path returns HTTP 404."""
        status, data = self._http_request("GET", "/non_existent_endpoint")
        self.assertEqual(status, 404)

    # 12. CORS preflight and headers
    def test_options_preflight_predict(self) -> None:
        """OPTIONS /predict returns HTTP 204 with CORS preflight headers for dashboard origin."""
        status, _, headers = self._http_request_raw(
            "OPTIONS",
            "/predict",
            headers={
                "Origin": "http://127.0.0.1:8501",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        self.assertEqual(status, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://127.0.0.1:8501")
        self.assertIn("POST", headers.get("Access-Control-Allow-Methods", ""))
        self.assertIn("OPTIONS", headers.get("Access-Control-Allow-Methods", ""))
        self.assertIn("Content-Type", headers.get("Access-Control-Allow-Headers", ""))

    def test_options_preflight_health(self) -> None:
        """OPTIONS /health returns HTTP 204 with CORS headers."""
        status, _, headers = self._http_request_raw(
            "OPTIONS",
            "/health",
            headers={"Origin": "http://127.0.0.1:8501"},
        )
        self.assertEqual(status, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://127.0.0.1:8501")
        self.assertIn("GET", headers.get("Access-Control-Allow-Methods", ""))

    def test_options_preflight_not_found(self) -> None:
        """OPTIONS to unknown path returns HTTP 404."""
        status, _, _ = self._http_request_raw(
            "OPTIONS",
            "/unknown_path",
            headers={"Origin": "http://127.0.0.1:8501"},
        )
        self.assertEqual(status, 404)

    def test_cors_headers_on_post_predict(self) -> None:
        """POST /predict response includes CORS headers matching allowed origin."""
        payload = {
            "restaurant_lat": 12.9716,
            "restaurant_lon": 77.5946,
            "customer_lat": 12.9816,
            "customer_lon": 77.6046,
            "order_hour": 19,
            "day_of_week": 4,
            "type_of_order": "Meal",
            "festival": "No",
            "city": "Metropolitian",
        }
        status, data, headers = self._http_request_raw(
            "POST",
            "/predict",
            headers={"Origin": "http://127.0.0.1:8501"},
            payload=payload,
        )
        self.assertEqual(status, 200)
        self.assertIn("predicted_duration_minutes", data)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://127.0.0.1:8501")
        self.assertIn("POST", headers.get("Access-Control-Allow-Methods", ""))

    def test_cors_headers_fallback_wildcard(self) -> None:
        """GET /health without Origin header provides wildcard CORS origin."""
        status, data, headers = self._http_request_raw("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")

