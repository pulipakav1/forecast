import os
import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_dir(path: str) -> str:
    if os.path.exists(path):
        return path
    alt = _PROJECT_ROOT / path
    return str(alt) if alt.exists() else path


def _series_dir(models_dir: str, series_id: str) -> str:
    safe = series_id.replace("/", "_")
    return os.path.join(models_dir, "m5", f"id={safe}")

def save_version(models_dir: str, series_id: str, model: Any, metadata: dict) -> dict:
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = _series_dir(models_dir, series_id)
    vdir = os.path.join(base, f"v{version}")
    os.makedirs(vdir, exist_ok=True)

    joblib.dump(model, os.path.join(vdir, "model.joblib"))

    metadata_out = {**metadata, "version": version, "series_id": series_id}
    with open(os.path.join(vdir, "metadata.json"), "w") as f:
        json.dump(metadata_out, f, indent=2)

    with open(os.path.join(base, "latest.json"), "w") as f:
        json.dump({"version": version}, f)

    return {"version": version, "path": vdir}

def load_latest(models_dir: str, series_id: str):
    models_dir = _resolve_dir(models_dir)
    base = _series_dir(models_dir, series_id)
    latest_path = os.path.join(base, "latest.json")
    if not os.path.exists(latest_path):
        raise FileNotFoundError(f"No latest model found for series_id={series_id}")

    with open(latest_path, "r") as f:
        latest = json.load(f)["version"]

    vdir = os.path.join(base, f"v{latest}")
    model = joblib.load(os.path.join(vdir, "model.joblib"))

    with open(os.path.join(vdir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    return model, metadata