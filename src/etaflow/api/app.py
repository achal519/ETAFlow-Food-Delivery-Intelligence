"""Standard library HTTP server application and request handler for ETAFlow."""

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from etaflow.api.schema import ETAPredictionRequest
from etaflow.api.service import ArtifactUnavailableError, ETAPredictionService
from etaflow.api.telemetry import APIMetricsTracker

logger = logging.getLogger(__name__)


ALLOWED_ORIGINS = {
    "http://127.0.0.1:8501",
    "http://localhost:8501",
}


class ETAPredictionHTTPHandler(BaseHTTPRequestHandler):
    """Standard-library HTTP request handler for ETAFlow Prediction API."""

    # Attached prediction service and telemetry tracker instances
    service: ETAPredictionService = None  # type: ignore[assignment]
    telemetry: APIMetricsTracker = APIMetricsTracker()

    def _send_cors_headers(self) -> None:
        """Send standard CORS headers allowing the local dashboard and required verbs."""
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        else:
            self.send_header("Access-Control-Allow-Origin", "*")

        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json_response(self, status: int, data: Dict[str, Any]) -> None:
        """Helper to serialize and write JSON response with appropriate headers."""
        try:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    REPORT_ENDPOINTS = {
        "/reports/experiments": "reports/experiments/experiment_summary.json",
        "/reports/drift": "reports/monitoring/drift_report.json",
        "/reports/reference-stats": "reports/monitoring/reference_stats.json",
        "/reports/validation": "reports/validation/data_validation_report.json",
        "/reports/registry": "models/registry.json",
    }

    def do_OPTIONS(self) -> None:
        """Handle OPTIONS preflight requests for CORS."""
        parsed_path = urlparse(self.path).path.rstrip("/")
        valid_endpoints = (
            "/predict",
            "/health",
            "/model-info",
            "/metrics",
            "",
            *self.REPORT_ENDPOINTS.keys(),
        )
        if parsed_path not in valid_endpoints:
            self._send_json_response(
                HTTPStatus.NOT_FOUND,
                {"error": f"Endpoint '{self.path}' not found."},
            )
            return

        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests for health probes, model registry info, telemetry metrics, and MLOps reports."""
        parsed_path = urlparse(self.path).path.rstrip("/")
        if parsed_path in ("", "/health"):
            if self.service and self.service.is_ready:
                self._send_json_response(
                    HTTPStatus.OK,
                    {
                        "status": "healthy",
                        "model_identity": self.service.model_identity,
                        "artifacts_loaded": True,
                    },
                )
                self.telemetry.record_request(HTTPStatus.OK)
            else:
                self._send_json_response(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "status": "unhealthy",
                        "error": "Model artifacts are not loaded or unavailable.",
                        "artifacts_loaded": False,
                    },
                )
                self.telemetry.record_request(HTTPStatus.SERVICE_UNAVAILABLE)
        elif parsed_path == "/model-info":
            if self.service:
                info = self.service.get_model_info()
                self._send_json_response(HTTPStatus.OK, info)
                self.telemetry.record_request(HTTPStatus.OK)
            else:
                self._send_json_response(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "Service is not initialized."},
                )
                self.telemetry.record_request(HTTPStatus.SERVICE_UNAVAILABLE)
        elif parsed_path == "/metrics":
            metrics = self.telemetry.get_metrics()
            self._send_json_response(HTTPStatus.OK, metrics)
            self.telemetry.record_request(HTTPStatus.OK)
        elif parsed_path in self.REPORT_ENDPOINTS:
            rel_file = self.REPORT_ENDPOINTS[parsed_path]
            project_root = Path(__file__).resolve().parents[3]
            target_path = project_root / rel_file
            if target_path.is_file():
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._send_json_response(HTTPStatus.OK, data)
                    self.telemetry.record_request(HTTPStatus.OK)
                except Exception as exc:
                    self._send_json_response(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": f"Failed reading {rel_file}: {exc}"},
                    )
            else:
                self._send_json_response(
                    HTTPStatus.NOT_FOUND,
                    {"error": f"Report file '{rel_file}' not found on server."},
                )
                self.telemetry.record_request(HTTPStatus.NOT_FOUND)
        else:
            self._send_json_response(
                HTTPStatus.NOT_FOUND,
                {"error": f"Endpoint '{self.path}' not found."},
            )
            self.telemetry.record_request(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        """Handle POST requests for model inference with real latency measurement."""
        t0 = time.perf_counter()
        parsed_path = urlparse(self.path).path.rstrip("/")
        if parsed_path != "/predict":
            self._send_json_response(
                HTTPStatus.NOT_FOUND,
                {"error": f"Endpoint '{self.path}' not found."},
            )
            self.telemetry.record_request(HTTPStatus.NOT_FOUND)
            return

        # Read incoming payload
        content_length_header = self.headers.get("Content-Length")
        if not content_length_header:
            self._send_json_response(
                HTTPStatus.BAD_REQUEST,
                {"error": "Missing Content-Length header or empty body."},
            )
            self.telemetry.record_request(HTTPStatus.BAD_REQUEST, is_predict=True)
            return

        try:
            content_length = int(content_length_header)
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json_response(
                HTTPStatus.BAD_REQUEST,
                {"error": "Invalid JSON payload.", "details": [str(exc)]},
            )
            self.telemetry.record_request(HTTPStatus.BAD_REQUEST, is_predict=True)
            return

        # Validate payload and reject forbidden post-booking fields
        request, errors = ETAPredictionRequest.from_dict(payload)
        if errors or request is None:
            self._send_json_response(
                HTTPStatus.BAD_REQUEST,
                {"error": "Validation failed", "details": errors},
            )
            self.telemetry.record_request(HTTPStatus.BAD_REQUEST, is_predict=True)
            return

        # Perform inference
        try:
            response = self.service.predict(request)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._send_json_response(HTTPStatus.OK, response.to_dict())
            self.telemetry.record_request(HTTPStatus.OK, latency_ms=latency_ms, is_predict=True)
        except ArtifactUnavailableError:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._send_json_response(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Prediction service is unavailable: Model artifacts are not loaded."},
            )
            self.telemetry.record_request(HTTPStatus.SERVICE_UNAVAILABLE, latency_ms=latency_ms, is_predict=True)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.error("Unexpected error during prediction: %s", exc, exc_info=False)
            self._send_json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "Internal prediction service error."},
            )
            self.telemetry.record_request(HTTPStatus.INTERNAL_SERVER_ERROR, latency_ms=latency_ms, is_predict=True)

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet default console logging; delegate to Python logging."""
        logger.debug("%s - - [%s] %s", self.client_address[0], self.log_date_time_string(), format % args)


def create_api_server(
    service: Optional[ETAPredictionService] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    telemetry: Optional[APIMetricsTracker] = None,
) -> HTTPServer:
    """Instantiate standard-library HTTPServer bound to the ETAPredictionHTTPHandler."""
    if service is None:
        service = ETAPredictionService()
    if telemetry is None:
        telemetry = APIMetricsTracker()

    class BoundHandler(ETAPredictionHTTPHandler):
        pass

    BoundHandler.service = service
    BoundHandler.telemetry = telemetry
    server = HTTPServer((host, port), BoundHandler)
    return server


def run_server(
    service: Optional[ETAPredictionService] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    telemetry: Optional[APIMetricsTracker] = None,
) -> None:
    """Run the prediction server indefinitely."""
    server = create_api_server(service=service, host=host, port=port, telemetry=telemetry)
    print(f"ETAFlow Prediction API serving at http://{host}:{port}/")
    print("Available endpoints:")
    print(f"  GET  http://{host}:{port}/health")
    print(f"  GET  http://{host}:{port}/model-info")
    print(f"  GET  http://{host}:{port}/metrics")
    print(f"  POST http://{host}:{port}/predict")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()
