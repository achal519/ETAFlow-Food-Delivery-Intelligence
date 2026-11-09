"""Unit tests for Phase 1 MLOps: Data validation, fingerprinting, registry, and telemetry."""

import json
from pathlib import Path
import threading
import time
import unittest
import urllib.request
from urllib.error import HTTPError
import pandas as pd

from etaflow.api.app import create_api_server
from etaflow.api.service import ETAPredictionService
from etaflow.api.telemetry import APIMetricsTracker
from etaflow.mlops.registry import ModelRegistry
from etaflow.mlops.validator import DatasetValidator, compute_file_sha256


class TestMLOpsPhase1(unittest.TestCase):
    """Test suite verifying data validation, SHA-256 fingerprinting, model registry, and telemetry."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start a test server on a free port with active telemetry and service."""
        cls.service = ETAPredictionService()
        cls.telemetry = APIMetricsTracker()
        cls.server = create_api_server(
            service=cls.service,
            host="127.0.0.1",
            port=0,
            telemetry=cls.telemetry,
        )
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
        headers: dict = None,
    ) -> tuple[int, dict, dict]:
        """Send HTTP request and return (status, parsed_body, response_headers)."""
        url = f"{self.base_url}{path}"
        data = None
        req_headers = dict(headers or {})
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            req_headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                body = resp.read().decode("utf-8")
                parsed = json.loads(body) if body else {}
                resp_headers = {k.title(): v for k, v in resp.headers.items()}
                return resp.status, parsed, resp_headers
        except HTTPError as exc:
            body = exc.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            exc_headers = {k.title(): v for k, v in exc.headers.items()}
            return exc.code, parsed, exc_headers

    # 1. SHA-256 Fingerprinting Tests
    def test_sha256_computation_and_nonexistent(self) -> None:
        """Verify deterministic hashing and missing file error handling."""
        raw_csv = Path("data/raw/zomato_cleaned.csv")
        self.assertTrue(raw_csv.is_file())
        digest = compute_file_sha256(raw_csv)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, "a76ed049cd698b189797b180ffce60dcdad7b45b99729cd171b5562ec585eb0c")

        with self.assertRaises(FileNotFoundError):
            compute_file_sha256("non_existent_file_xyz.csv")

    # 2. Data Validation Tests
    def test_dataset_validator_on_raw_csv(self) -> None:
        """Verify DatasetValidator runs successfully on the repository raw dataset."""
        validator = DatasetValidator()
        report = validator.validate_file("data/raw/zomato_cleaned.csv")

        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_rows, 38964)
        self.assertEqual(report.total_columns, 22)
        self.assertEqual(report.dataset_sha256, "a76ed049cd698b189797b180ffce60dcdad7b45b99729cd171b5562ec585eb0c")
        self.assertEqual(len(report.schema_errors), 0)
        self.assertEqual(len(report.range_errors), 0)
        self.assertEqual(len(report.category_errors), 0)

    def test_dataset_validator_failure_cases(self) -> None:
        """Verify validator flags missing columns, invalid ranges, and invalid categories."""
        bad_df = pd.DataFrame({
            # Restaurant_latitude deliberately omitted to test missing column detection
            "Restaurant_longitude": [77.5],
            "Delivery_location_latitude": [999.0],  # Out of latitude bounds
            "Delivery_location_longitude": [77.6],
            "Type_of_order": ["PizzaBurger"],  # Unrecognized category
            "Festival": ["UnknownFestival"],    # Unrecognized category
            "City": ["Atlantis"],               # Unrecognized category
            "Time_taken (min)": [-5.0],         # Negative duration
        })

        validator = DatasetValidator()
        report = validator.validate_dataframe(bad_df, dataset_path="test_mock")
        self.assertFalse(report.is_valid)
        self.assertTrue(any("Missing required columns" in e for e in report.schema_errors))
        self.assertTrue(any("Target values out of valid range" in e for e in report.range_errors))
        self.assertTrue(any("Latitude column" in e for e in report.range_errors))
        self.assertTrue(any("Type_of_order" in e for e in report.category_errors))
        self.assertTrue(any("Festival" in e for e in report.category_errors))
        self.assertTrue(any("City" in e for e in report.category_errors))

    def test_validation_report_json_export(self) -> None:
        """Verify ValidationReport exports to valid JSON and reloads intact."""
        dest_path = Path("reports/validation/data_validation_report.json")
        self.assertTrue(dest_path.is_file())
        with open(dest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertTrue(data["is_valid"])
        self.assertEqual(data["total_rows"], 38964)
        self.assertEqual(data["dataset_sha256"], "a76ed049cd698b189797b180ffce60dcdad7b45b99729cd171b5562ec585eb0c")

    # 3. Model Registry Tests
    def test_model_registry_manifest_contents(self) -> None:
        """Verify ModelRegistry loads versions and identifies the active champion."""
        reg = ModelRegistry("models/registry.json")
        self.assertEqual(reg.active_version, "v1.0.0")

        champion = reg.get_active_model()
        self.assertIsNotNone(champion)
        self.assertEqual(champion.version, "v1.0.0")
        self.assertEqual(champion.model_identity, "random_forest_regressor")
        self.assertEqual(champion.stage, "Production")
        self.assertTrue(champion.is_active)
        self.assertEqual(champion.test_rows, 7793)
        self.assertEqual(champion.evaluation_metrics["mae"], 6.1406)
        self.assertEqual(champion.evaluation_metrics["r2"], 0.3036)

        # Baseline versions also present in history
        self.assertIn("v0.1.0-dummy", reg.versions)
        self.assertIn("v0.2.0-ridge", reg.versions)

    def test_model_registry_artifact_integrity(self) -> None:
        """Verify registry cryptographic verification passes for v1.0.0."""
        reg = ModelRegistry("models/registry.json")
        integrity = reg.verify_integrity("v1.0.0")
        self.assertTrue(integrity["model_artifact"])
        self.assertTrue(integrity["preprocessor_artifact"])

    # 4. Telemetry and API Endpoint Tests
    def test_api_model_info_endpoint(self) -> None:
        """Verify GET /model-info returns 200 with registered model metadata and hashes."""
        status, body, headers = self._http_request("GET", "/model-info")
        self.assertEqual(status, 200)
        self.assertEqual(body["model_identity"], "random_forest_regressor")
        self.assertEqual(body["version"], "v1.0.0")
        self.assertEqual(body["stage"], "Production")
        self.assertTrue(body["artifacts_loaded"])
        self.assertEqual(body["evaluation_test_rows"], 7793)
        self.assertEqual(body["evaluation_metrics"]["mae"], 6.1406)
        self.assertIsNotNone(body["model_artifact"]["sha256_hash"])
        self.assertIsNotNone(body["preprocessor_artifact"]["sha256_hash"])
        self.assertEqual(len(body["model_artifact"]["sha256_hash"]), 64)

    def test_api_metrics_endpoint_and_timing(self) -> None:
        """Verify GET /metrics exposes real request counters and measured latencies."""
        # Check initial metrics
        status, initial_metrics, _ = self._http_request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertIn("uptime_seconds", initial_metrics)
        self.assertIn("latency_ms", initial_metrics)
        prev_successful = initial_metrics["successful_predictions"]

        # Run a valid predict request
        payload = {
            "restaurant_lat": 12.9352,
            "restaurant_lon": 77.6245,
            "customer_lat": 12.9550,
            "customer_lon": 77.6400,
            "order_hour": 20,
            "day_of_week": 4,
            "type_of_order": "Meal",
            "festival": "No",
            "city": "Metropolitian",
            "order_id": "TEST-TELEMETRY-001",
        }
        pred_status, pred_body, _ = self._http_request("POST", "/predict", payload=payload)
        self.assertEqual(pred_status, 200)
        self.assertIn("predicted_duration_minutes", pred_body)

        # Run an invalid request
        inv_status, _, _ = self._http_request("POST", "/predict", payload={"invalid": "payload"})
        self.assertEqual(inv_status, 400)

        # Check updated metrics
        _, updated_metrics, _ = self._http_request("GET", "/metrics")
        self.assertEqual(updated_metrics["successful_predictions"], prev_successful + 1)
        self.assertGreaterEqual(updated_metrics["failed_predictions"], 1)
        self.assertGreater(updated_metrics["latency_ms"]["latest"], 0.0)
        self.assertGreater(updated_metrics["latency_ms"]["p50"], 0.0)

    def test_api_options_preflight_for_new_endpoints(self) -> None:
        """Verify OPTIONS preflight returns HTTP 204 for /model-info and /metrics."""
        for endpoint in ["/model-info", "/metrics"]:
            status, _, headers = self._http_request(
                "OPTIONS",
                endpoint,
                headers={"Origin": "http://127.0.0.1:8501"},
            )
            self.assertEqual(status, 204)
            self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://127.0.0.1:8501")
            self.assertIn("GET", headers.get("Access-Control-Allow-Methods", ""))
