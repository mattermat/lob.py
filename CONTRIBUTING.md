# Contributing

## Development Setup

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
pip install -e .
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

## Running Tests

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=lobpy
```

## Code Quality

### Linting
```bash
ruff check .
```

### Formatting
```bash
black .
```

### Type Checking
```bash
mypy lobpy
```

### Security Scanning
```bash
bandit -r lobpy
```

## Pre-commit Hooks

The pre-commit hooks will automatically run:
- Black formatting
- Ruff linting
- Trailing whitespace removal
- YAML file validation

## Publishing a New Release

1. Update the version in `pyproject.toml`
2. Create a git tag:
```bash
git tag v0.x.x
git push origin v0.x.x
```

The GitHub Actions workflow will automatically publish to PyPI.
