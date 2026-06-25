.PHONY: help install dev lint format test test-all coverage clean build publish

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

dev:  ## Install with dev dependencies
	pip install -e ".[dev]"
	pre-commit install

lint:  ## Run linters
	ruff check src/ tests/
	black --check src/ tests/
	mypy src/dspy_guardrails/ --ignore-missing-imports

format:  ## Auto-format code
	ruff check --fix src/ tests/
	black src/ tests/

test:  ## Run fast tests
	pytest tests/ -m "not llm and not slow" -q

test-all:  ## Run all tests including slow
	pytest tests/ -q

coverage:  ## Run tests with coverage report
	pytest tests/ -m "not llm and not slow" --cov=dspy_guardrails --cov-report=term-missing --cov-report=html

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/ .coverage coverage.xml

build:  ## Build package
	python -m build

publish: build  ## Publish to PyPI (requires credentials)
	twine upload dist/*
