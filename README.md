# Predictive Linear Regression Model Pipeline

[![CI](https://github.com/Belloblaze-web3/predictive-linear-regression/actions/workflows/ci.yml/badge.svg)](https://github.com/Belloblaze-web3/predictive-linear-regression/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-style supervised regression project using the real **Auto MPG** dataset from OpenML. The project demonstrates a leakage-safe scikit-learn workflow with:

- Mean imputation for missing numeric values
- Most-frequent imputation for missing categorical values
- One-hot encoding for categorical features
- Standard feature scaling
- Ridge regression regularization
- Reproducible train/test splitting
- Cross-validated hyperparameter selection
- Logged evaluation metrics: R² and RMSE
- Saved model artifact and metrics JSON

## Project layout

```text
predictive-linear-regression/
├── configs/config.yaml
├── data/                         # generated locally; ignored by git
├── models/                       # generated model artifact; ignored by git
├── reports/                      # generated metrics and plots; ignored by git
├── src/predictive_regression/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   ├── predict.py
│   └── train.py
├── tests/test_pipeline.py
├── tests/test_prediction.py
├── app.py                       # Streamlit interactive prediction UI
├── requirements-streamlit.txt
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE
└── .github/workflows/ci.yml
```

## Quick start

```bash
cd predictive-linear-regression
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m predictive_regression.train
pytest
```

## Run the interactive Streamlit app

Install the app extra and launch the local web application:

```bash
pip install -e '.[app]'
streamlit run app.py
```

Open the URL printed by Streamlit, enter vehicle specifications, and select **Predict fuel economy**. The app loads `models/ridge_pipeline.joblib` when available. If the artifact is absent—as it may be on a fresh deployment—the app trains the configured pipeline once from OpenML, caches the resulting model for the session, and then serves predictions.

For Streamlit Community Cloud, set the repository to this project, use `app.py` as the main file, and use `requirements-streamlit.txt` as the dependency file. The deployment needs outbound access to OpenML only when the saved model artifact is not included.

To run the same app in Docker:

```bash
docker build -t auto-mpg-predictor .
docker run --rm -p 8501:8501 auto-mpg-predictor
```

The training command downloads the dataset through scikit-learn/OpenML on the first run, trains the model, and writes:

- `models/ridge_pipeline.joblib` — complete preprocessing + estimator pipeline
- `reports/metrics.json` — dataset summary, configuration, and test metrics
- `reports/predictions.csv` — held-out predictions and residuals
- `reports/residuals.png` — diagnostic plot

## Modeling approach

The target is `mpg` (miles per gallon). Numeric features include displacement, horsepower, weight, acceleration, model year, and cylinders. `origin` is treated as a categorical feature and one-hot encoded, ensuring the pipeline exercises categorical preprocessing on the real dataset.

The pipeline is fit only on the training split. A `GridSearchCV` over Ridge `alpha` values selects regularization strength using 5-fold cross-validation and negative RMSE as the selection objective. Final performance is reported on a held-out test set that is never used during model selection.

## Reproducibility and quality

- Fixed random seed (`42`) for the train/test split and cross-validation.
- `Pipeline` and `ColumnTransformer` prevent preprocessing leakage.
- Type hints, logging, and explicit configuration are included.
- Tests cover dataset shape, preprocessing behavior, and end-to-end training.
- The project is designed to be extended with experiment tracking, CI, or a model registry.

## Latest validated run

Using the configured OpenML Auto MPG dataset, random seed `42`, 80/20 holdout, and 5-fold Ridge alpha search:

| Metric | Value |
|---|---:|
| Test R² | 0.8448 |
| Test RMSE | 2.8882 MPG |
| Selected alpha | 1.0 |

These values are logged to `reports/metrics.json` after training. Generated reports and model binaries are intentionally excluded from source control; the training command recreates them deterministically.

## Example output

The exact values depend on the current OpenML copy and split, but the command reports metrics in this form:

```text
Test R²:   0.84
Test RMSE: 3.10 mpg
Best alpha: 10.0
```

The example above is illustrative; use `reports/metrics.json` for the run's actual values.

## License and contribution

This project is released under the [MIT License](LICENSE). See [CONTRIBUTING.md](CONTRIBUTING.md) for development and pull-request standards.
