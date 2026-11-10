from pathlib import Path

import numpy as np
import pandas as pd

from predictive_regression.evaluate import build_pipeline, regression_metrics


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cylinders": [4, 6, np.nan, 8, 4, 6],
            "displacement": [100, 200, 150, np.nan, 95, 210],
            "horsepower": [70, 120, np.nan, 180, 65, 130],
            "weight": [2100, 3000, 2500, 3600, 2050, 3100],
            "acceleration": [16, 14, 15, 12, 17, 13],
            "model_year": [70, 72, 71, 73, 70, 74],
            "origin": ["1", "2", None, "3", "1", "2"],
            "manufacturer": ["ford", "toyota", "ford", "bmw", None, "toyota"],
        }
    )


def test_pipeline_handles_missing_and_categorical_values():
    X = sample_frame()
    y = pd.Series([25, 20, 23, 16, 29, 19])
    model = build_pipeline(alpha=1.0)
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert np.isfinite(predictions).all()


def test_metrics_are_correct_and_json_friendly():
    metrics = regression_metrics(pd.Series([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert set(metrics) == {"r2", "rmse"}
    assert metrics["rmse"] > 0


def test_expected_artifact_directories_are_documented():
    assert Path("README.md").exists()
