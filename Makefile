# Orchestry Development Makefile
.PHONY: help install dev-install format lint type-check test test-cov clean run-marl run-legacy docs

# Default target
.DEFAULT_GOAL := help

help:  ## Show this help message
	@echo "Orchestry Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install:  ## Install package
	uv pip install -e .

dev-install:  ## Install package with dev dependencies
	uv pip install -e ".[dev]"
	pre-commit install

# Code Quality
format:  ## Format code with Black and Ruff
	black orchestry/ examples/ tests/
	ruff check --fix --unsafe-fixes orchestry/ examples/ tests/ || true
	ruff format orchestry/ examples/ tests/

lint:  ## Run linting with Ruff
	ruff check orchestry/ examples/ tests/

type-check:  ## Run type checking with mypy
	mypy orchestry/

check: lint type-check  ## Run all checks (lint + type-check)

# Testing
test:  ## Run tests
	pytest tests/

test-cov:  ## Run tests with coverage
	pytest --cov=orchestry --cov-report=html --cov-report=term

test-fast:  ## Run tests in parallel
	pytest -n auto tests/

# Cleaning
clean:  ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean  ## Clean everything including venv
	rm -rf .venv/

# Running
run-marl:  ## Run MARL training (dry-run mode)
	python -m orchestry.cli.marl --dry-run --verbose

run-legacy:  ## Run legacy training
	python -m orchestry.cli.legacy --test --verbose

# Development
pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

lock:  ## Update uv.lock file
	uv lock

sync:  ## Sync dependencies from lock file
	uv sync

# Documentation
docs:  ## Open documentation
	@echo "Documentation is in docs/ directory"
	@echo "Main README: README.md"
	@ls -1 docs/

# CI/CD simulation
ci: format check test  ## Run full CI pipeline locally

# Quick checks before commit
quick: format lint  ## Quick format and lint before commit

# All quality checks
all: format lint type-check test  ## Run all quality checks
