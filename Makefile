.PHONY: install test train clean

install:
	python3 -m pip install -e '.[dev]'

test:
	pytest -q

train:
	python -m predictive_regression.train

clean:
	rm -rf .pytest_cache .coverage htmlcov
