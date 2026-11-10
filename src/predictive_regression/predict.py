"""Inference helpers for the interactive Streamlit application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load

from .train import train

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def _artifact_path(config_path: Path, key: str) -> Path:
    """Resolve a configured artifact path relative to the project root."""
    import yaml

    config = yaml.safe_load(config_path.read_text())
    path = Path(config["artifacts"][key])
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_or_train_model(config_path: str | Path = DEFAULT_CONFIG) -> tuple[Any, dict]:
    """Load the saved pipeline, training it once if it is not available."""
    config_path = Path(config_path)
    model_path = _artifact_path(config_path, "model_path")
    metrics_path = _artifact_path(config_path, "metrics_path")
    if not model_path.exists():
        train(config_path)
    model = load(model_path)
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return model, metrics


def predict_mpg(model: Any, features: dict[str, Any]) -> float:
    """Predict MPG for one validated feature dictionary."""
    frame = pd.DataFrame([features])
    return float(model.predict(frame)[0])
