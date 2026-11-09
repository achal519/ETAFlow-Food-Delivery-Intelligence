"""Model registry and versioning manifest manager for ETAFlow."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from etaflow.mlops.validator import compute_file_sha256


@dataclass
class ArtifactMetadata:
    """Cryptographic and file metadata for serialized artifacts."""

    artifact_name: str
    relative_path: str
    sha256_hash: str
    size_bytes: int


@dataclass
class ModelVersionManifest:
    """Complete specification of a registered model version."""

    version: str
    model_identity: str
    algorithm: str
    stage: str  # "Production", "Staging", "Archived"
    is_active: bool
    registration_timestamp: str
    dataset_fingerprint: str
    training_rows: int
    test_rows: int
    feature_columns: List[str]
    encoded_feature_count: int
    hyperparameters: Dict[str, Any]
    evaluation_metrics: Dict[str, float]
    artifacts: Dict[str, Dict[str, Any]]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        return asdict(self)


class ModelRegistry:
    """Persistent JSON manifest model registry for versioning and verification."""

    def __init__(self, registry_file: Union[str, Path] = "models/registry.json") -> None:
        self.registry_file = Path(registry_file)
        self.versions: Dict[str, ModelVersionManifest] = {}
        self.active_version: Optional[str] = None
        if self.registry_file.is_file():
            self.load()

    def load(self) -> None:
        """Load registered versions from registry JSON file."""
        with open(self.registry_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.active_version = data.get("active_version")
        self.versions = {}
        for ver_str, ver_data in data.get("versions", {}).items():
            manifest = ModelVersionManifest(
                version=ver_data["version"],
                model_identity=ver_data["model_identity"],
                algorithm=ver_data["algorithm"],
                stage=ver_data["stage"],
                is_active=ver_data["is_active"],
                registration_timestamp=ver_data["registration_timestamp"],
                dataset_fingerprint=ver_data["dataset_fingerprint"],
                training_rows=ver_data["training_rows"],
                test_rows=ver_data["test_rows"],
                feature_columns=ver_data["feature_columns"],
                encoded_feature_count=ver_data["encoded_feature_count"],
                hyperparameters=ver_data["hyperparameters"],
                evaluation_metrics=ver_data["evaluation_metrics"],
                artifacts=ver_data["artifacts"],
                notes=ver_data.get("notes", ""),
            )
            self.versions[ver_str] = manifest

    def save(self) -> Path:
        """Persist registry state to disk."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_version": self.active_version,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
        }
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return self.registry_file

    def get_version(self, version: str) -> Optional[ModelVersionManifest]:
        """Retrieve a specific registered version by tag."""
        return self.versions.get(version)

    def get_active_model(self) -> Optional[ModelVersionManifest]:
        """Return the currently served active production model manifest."""
        if self.active_version and self.active_version in self.versions:
            return self.versions[self.active_version]
        for manifest in self.versions.values():
            if manifest.is_active:
                return manifest
        return None

    def list_versions(self) -> List[Dict[str, Any]]:
        """List summary of all registered versions."""
        return [v.to_dict() for v in self.versions.values()]

    def register_version(
        self,
        version: str,
        model_identity: str,
        algorithm: str,
        stage: str,
        dataset_fingerprint: str,
        training_rows: int,
        test_rows: int,
        feature_columns: List[str],
        encoded_feature_count: int,
        hyperparameters: Dict[str, Any],
        evaluation_metrics: Dict[str, float],
        artifacts: Dict[str, Union[str, Path]],
        is_active: bool = False,
        notes: str = "",
        project_root: Optional[Union[str, Path]] = None,
    ) -> ModelVersionManifest:
        """Register a new model version with automatic SHA-256 fingerprinting of artifacts."""
        root = Path(project_root) if project_root else Path(".")
        artifact_metas: Dict[str, Dict[str, Any]] = {}

        for art_key, art_path in artifacts.items():
            full_path = root / art_path if not Path(art_path).is_absolute() else Path(art_path)
            if not full_path.is_file():
                raise FileNotFoundError(f"Cannot register artifact: file not found at {full_path}")

            sha256 = compute_file_sha256(full_path)
            size = full_path.stat().st_size
            rel_path = str(full_path.relative_to(root)) if full_path.is_relative_to(root) else str(full_path)
            artifact_metas[art_key] = {
                "artifact_name": art_key,
                "relative_path": rel_path.replace("\\", "/"),
                "sha256_hash": sha256,
                "size_bytes": size,
            }

        if is_active:
            for v in self.versions.values():
                v.is_active = False
            self.active_version = version

        manifest = ModelVersionManifest(
            version=version,
            model_identity=model_identity,
            algorithm=algorithm,
            stage=stage,
            is_active=is_active,
            registration_timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_fingerprint=dataset_fingerprint,
            training_rows=training_rows,
            test_rows=test_rows,
            feature_columns=feature_columns,
            encoded_feature_count=encoded_feature_count,
            hyperparameters=hyperparameters,
            evaluation_metrics=evaluation_metrics,
            artifacts=artifact_metas,
            notes=notes,
        )

        self.versions[version] = manifest
        self.save()
        return manifest

    def verify_integrity(self, version: str, project_root: Optional[Union[str, Path]] = None) -> Dict[str, bool]:
        """Verify on-disk artifact integrity against registered SHA-256 digests.

        Returns:
            Dictionary mapping artifact_name -> True if matches, False if altered or missing.
        """
        manifest = self.get_version(version)
        if not manifest:
            raise KeyError(f"Version '{version}' is not registered.")

        root = Path(project_root) if project_root else Path(".")
        results: Dict[str, bool] = {}
        for name, meta in manifest.artifacts.items():
            target = root / meta["relative_path"]
            if not target.is_file():
                results[name] = False
                continue
            current_hash = compute_file_sha256(target)
            results[name] = current_hash == meta["sha256_hash"]
        return results
