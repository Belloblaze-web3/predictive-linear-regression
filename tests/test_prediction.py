import pandas as pd

from predictive_regression.evaluate import build_pipeline
from predictive_regression.predict import predict_mpg


def test_predict_mpg_returns_scalar_prediction():
    features = pd.DataFrame(
        {
            "cylinders": [4, 6, 8],
            "displacement": [100.0, 200.0, 350.0],
            "horsepower": [70.0, 120.0, 180.0],
            "weight": [2100.0, 3000.0, 3600.0],
            "acceleration": [16.0, 14.0, 12.0],
            "model_year": [70, 72, 76],
            "origin": ["1", "2", "3"],
        }
    )
    target = pd.Series([28.0, 21.0, 16.0])
    model = build_pipeline().fit(features, target)
    prediction = predict_mpg(model, features.iloc[0].to_dict())
    assert isinstance(prediction, float)
    assert prediction > 0
