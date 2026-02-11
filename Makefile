.PHONY: help install test lint format typecheck security clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install development dependencies and package
	pip install -r requirements-dev.txt
	pip install -e .

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=lobpy --cov-report=html

lint: ## Run linter
	ruff check .

format: ## Format code with black
	black .

format-check: ## Check code formatting
	black --check .

typecheck: ## Run type checker
	mypy lobpy

security: ## Run security scanner
	bandit -r lobpy

pre-commit: ## Install pre-commit hooks
	pre-commit install

check: lint format-check typecheck security test ## Run all checks

clean: ## Clean up build artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
