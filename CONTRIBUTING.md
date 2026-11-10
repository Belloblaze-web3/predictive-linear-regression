# Contributing

Thanks for contributing. Please open an issue before substantial changes so the proposed modeling or data changes can be discussed.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

## Pull requests

Pull requests should include a concise problem statement, test coverage, and any changes to evaluation metrics or data assumptions. Keep preprocessing inside the scikit-learn pipeline to avoid leakage. Do not commit datasets, model binaries, credentials, or generated reports unless explicitly required for a release.

All pull requests must pass the GitHub Actions test matrix before merging.
