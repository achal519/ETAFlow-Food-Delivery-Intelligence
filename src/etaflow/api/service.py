"""ETAPredictionService: loads fitted artifacts and executes single-order prediction."""

import logging
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd

from etaflow.api.schema import ETAPredictionRequest, ETAPredictionResponse
from etaflow.features.pipeline import calculate_haversine_distance

logger = logging.getLogger(__name__)

# Exact 11 Tier 1 booking-time features expected by preprocessor.joblib
PREPROCESSOR_INPUT_COLUMNS = [
    "distance_km",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "order_hour",
    "day_of_week",
    "is_weekend",
    "Type_of_order",
    "Festival",
    "City",
]


class ArtifactUnavailableError(Exception):
    """Raised when model or preprocessor artifacts cannot be loaded."""


class ETAPredictionService:
    """Service encapsulating artifact loading, feature mapping, and inference."""

    def __init__(
        self,
        preprocessor_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        model_identity: str = "random_forest_regressor",
    ) -> None:
        """Initialize the prediction service with optional artifact paths.

        Paths default to resolving relative to the project root.
        """
        # Resolve project root (4 levels up from this file: src/etaflow/api/service.py -> root)
        project_root = Path(__file__).resolve().parents[3]

        self.project_root = project_root
        self.preprocessor_path = preprocessor_path or (project_root / "models" / "preprocessor.joblib")
        self.model_path = model_path or (project_root / "models" / "random_forest_regressor.joblib")
        self.model_identity = model_identity

        self.preprocessor = None
        self.model = None
        self.model_sha256: Optional[str] = None
        self.preprocessor_sha256: Optional[str] = None
        self.is_ready = False
        self.load_error: Optional[str] = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load preprocessor and model artifacts from disk and compute cryptographic hashes."""
        try:
            if not self.preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor artifact not found at {self.preprocessor_path.name}")
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model artifact not found at {self.model_path.name}")

            from etaflow.mlops.validator import compute_file_sha256

            self.preprocessor = joblib.load(self.preprocessor_path)
            self.model = joblib.load(self.model_path)
            self.model_sha256 = compute_file_sha256(self.model_path)
            self.preprocessor_sha256 = compute_file_sha256(self.preprocessor_path)
            self.is_ready = True
            self.load_error = None
            logger.info("ETAPredictionService successfully loaded artifacts.")
        except Exception as exc:
            self.preprocessor = None
            self.model = None
            self.model_sha256 = None
            self.preprocessor_sha256 = None
            self.is_ready = False
            # Store sanitized error message without exposing full internal paths
            self.load_error = f"Artifact loading error: {exc.__class__.__name__}"
            logger.warning("ETAPredictionService failed to load artifacts: %s", exc)

    def get_model_info(self) -> dict:
        """Return metadata, versioning, and cryptographic fingerprints for the served model."""
        import json

        registry_file = self.project_root / "models" / "registry.json"
        version_tag = "v1.0.0"
        stage = "Production"
        metrics = {
            "mae": 6.1406,
            "rmse": 7.6987,
            "r2": 0.3036,
            "mape": 27.41,
        }
        dataset_fingerprint = ""

        if registry_file.is_file():
            try:
                with open(registry_file, "r", encoding="utf-8") as f:
                    reg_data = json.load(f)
                active_ver = reg_data.get("active_version", "v1.0.0")
                ver_info = reg_data.get("versions", {}).get(active_ver, {})
                if ver_info:
                    version_tag = ver_info.get("version", version_tag)
                    stage = ver_info.get("stage", stage)
                    metrics = ver_info.get("evaluation_metrics", metrics)
                    dataset_fingerprint = ver_info.get("dataset_fingerprint", "")
            except Exception as e:
                logger.debug("Could not read registry file: %s", e)

        return {
            "model_identity": self.model_identity,
            "version": version_tag,
            "stage": stage,
            "artifacts_loaded": self.is_ready,
            "model_artifact": {
                "name": self.model_path.name,
                "sha256_hash": self.model_sha256,
            },
            "preprocessor_artifact": {
                "name": self.preprocessor_path.name,
                "sha256_hash": self.preprocessor_sha256,
            },
            "dataset_fingerprint": dataset_fingerprint,
            "feature_columns": PREPROCESSOR_INPUT_COLUMNS,
            "evaluation_metrics": metrics,
            "evaluation_test_rows": 7793,
        }

    def prepare_feature_dataframe(self, request: ETAPredictionRequest) -> pd.DataFrame:
        """Map ETAPredictionRequest into 1-row DataFrame matching the 11 feature schema."""
        distance_km = calculate_haversine_distance(
            lat1=request.restaurant_lat,
            lon1=request.restaurant_lon,
            lat2=request.customer_lat,
            lon2=request.customer_lon,
        )
        is_weekend = 1 if request.day_of_week in (5, 6) else 0

        row = {
            "distance_km": float(distance_km),
            "Restaurant_latitude": float(request.restaurant_lat),
            "Restaurant_longitude": float(request.restaurant_lon),
            "Delivery_location_latitude": float(request.customer_lat),
            "Delivery_location_longitude": float(request.customer_lon),
            "order_hour": int(request.order_hour),
            "day_of_week": int(request.day_of_week),
            "is_weekend": int(is_weekend),
            "Type_of_order": str(request.type_of_order),
            "Festival": str(request.festival),
            "City": str(request.city),
        }

        # Build DataFrame with the exact column order
        return pd.DataFrame([row], columns=PREPROCESSOR_INPUT_COLUMNS)

    def predict(self, request: ETAPredictionRequest) -> ETAPredictionResponse:
        """Execute end-to-end prediction for an incoming booking request.

        Raises:
            ArtifactUnavailableError: If model or preprocessor artifacts are not loaded.
        """
        if not self.is_ready or self.preprocessor is None or self.model is None:
            raise ArtifactUnavailableError(
                "Prediction service is unavailable: Model or preprocessor artifact is not loaded."
            )

        df_11 = self.prepare_feature_dataframe(request)
        distance_km = float(df_11["distance_km"].iloc[0])

        # Preprocess features into the 17 encoded columns
        transformed = self.preprocessor.transform(df_11)
        feature_names = self.preprocessor.get_feature_names_out()
        df_17 = pd.DataFrame(transformed, columns=feature_names)

        # Run inference through Random Forest
        raw_prediction = float(self.model.predict(df_17)[0])

        return ETAPredictionResponse(
            predicted_duration_minutes=raw_prediction,
            model_identity=self.model_identity,
            distance_km=distance_km,
            order_id=request.order_id,
            prediction_stage="booking_time",
        )
