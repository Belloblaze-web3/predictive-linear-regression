"""Train and evaluate the Auto MPG Ridge model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from joblib import dump
from sklearn.model_selection import GridSearchCV, train_test_split

from .data import load_auto_mpg, split_features_target
from .evaluate import build_pipeline, regression_metrics

LOGGER = logging.getLogger(__name__)


def train(config_path: str | Path = "configs/config.yaml") -> dict:
    """Train the model and write reproducible evaluation artifacts."""
    config = yaml.safe_load(Path(config_path).read_text())
    data_cfg, model_cfg, artifact_cfg = config["data"], config["model"], config["artifacts"]
    for key in artifact_cfg.values():
        Path(key).parent.mkdir(parents=True, exist_ok=True)

    frame = load_auto_mpg(data_cfg["openml_name"], data_cfg["openml_version"])
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"]
    )

    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid={"regressor__alpha": model_cfg["ridge_alphas"]},
        cv=model_cfg["cv_folds"],
        scoring=model_cfg["scoring"],
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)
    metrics = regression_metrics(y_test, predictions)

    prediction_frame = pd.DataFrame(
        {"actual_mpg": y_test.to_numpy(), "predicted_mpg": predictions}
    )
    prediction_frame["residual"] = prediction_frame["actual_mpg"] - prediction_frame["predicted_mpg"]
    prediction_frame.to_csv(artifact_cfg["predictions_path"], index=False)
    dump(search.best_estimator_, artifact_cfg["model_path"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(prediction_frame["predicted_mpg"], prediction_frame["residual"], alpha=0.75)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Auto MPG Ridge residuals", xlabel="Predicted MPG", ylabel="Residual (actual − predicted)")
    fig.tight_layout()
    fig.savefig(artifact_cfg["plot_path"], dpi=160)
    plt.close(fig)

    result = {
        "dataset": {"name": data_cfg["openml_name"], "rows": int(len(frame)), "features": int(X.shape[1])},
        "split": {"test_size": data_cfg["test_size"], "random_state": data_cfg["random_state"], "test_rows": int(len(X_test))},
        "model": {"estimator": "Ridge", "best_alpha": float(search.best_params_["regressor__alpha"]), "cv_folds": model_cfg["cv_folds"]},
        "test_metrics": metrics,
    }
    Path(artifact_cfg["metrics_path"]).write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("Test R²: %.4f | Test RMSE: %.4f mpg | Best alpha: %s", metrics["r2"], metrics["rmse"], result["model"]["best_alpha"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = train(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
