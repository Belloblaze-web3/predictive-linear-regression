"""Data ingestion and feature engineering."""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_openml

LOGGER = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
]
CATEGORICAL_FEATURES = ["origin"]


def load_auto_mpg(name: str = "autoMpg", version: int = 1) -> pd.DataFrame:
    """Fetch the real Auto MPG dataset from OpenML and normalize its schema."""
    # ``liac-arff`` handles the mixed numeric/categorical OpenML schema more
    # robustly than the newer pandas parser for this dataset's integer-valued
    # categorical levels.
    bunch = fetch_openml(name=name, version=version, as_frame=True, parser="liac-arff")
    frame = bunch.frame.copy()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]

    # In this OpenML version, scikit-learn exposes the target separately as
    # ``bunch.target`` and names model year simply ``model``.
    frame["mpg"] = pd.to_numeric(bunch.target, errors="coerce").to_numpy()
    if "model" in frame.columns:
        frame = frame.rename(columns={"model": "model_year"})
    elif "model_year" not in frame.columns:
        raise ValueError(f"Expected model-year column; found {sorted(frame.columns)}")
    frame["origin"] = frame["origin"].astype("string")
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    selected = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["mpg"]
    missing = sorted(set(selected) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")
    result = frame[selected].dropna(subset=["mpg"]).reset_index(drop=True)
    LOGGER.info("Loaded %d rows with %d features from OpenML/%s", len(result), len(selected) - 1, name)
    return result


def split_features_target(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix and continuous target."""
    return frame.drop(columns="mpg"), frame["mpg"]
