"""CLI runner to serve the ETAFlow prediction API locally."""

import argparse
from pathlib import Path
import sys
import yaml

from etaflow.api.app import run_server
from etaflow.api.service import ETAPredictionService


def main() -> None:
    """Parse CLI arguments, load configuration, and start prediction server."""
    parser = argparse.ArgumentParser(description="Serve the ETAFlow Prediction API locally.")
    parser.add_argument("--host", type=str, default=None, help="Host address to bind (default: from config or 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: from config or 8000)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration YAML")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / args.config

    host = "127.0.0.1"
    port = 8000
    model_name = "random_forest_regressor"

    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            api_cfg = cfg.get("api", {})
            host = api_cfg.get("host", host)
            port = int(api_cfg.get("port", port))
            model_name = api_cfg.get("model_name", model_name)
        except Exception as exc:
            print(f"Warning: Could not read config file ({exc}). Falling back to defaults.")

    if args.host:
        host = args.host
    if args.port:
        port = args.port

    print("Initializing ETAFlow Prediction Service...")
    service = ETAPredictionService(model_identity=model_name)
    if not service.is_ready:
        print(f"Warning: {service.load_error}. Service will return 503 until artifacts are present.", file=sys.stderr)
    else:
        print(f"Ready: Model '{service.model_identity}' and preprocessor successfully loaded.")

    run_server(service=service, host=host, port=port)


if __name__ == "__main__":
    main()
