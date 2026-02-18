# Agent Instructions

## Virtual Environment

This project uses a virtual environment located in `.env/`.

**IMPORTANT**: Always activate the virtual environment before running Python commands:

```bash
source .env/bin/activate
```

### Running Tests

Always run tests with the venv activated:

```bash
source .env/bin/activate && pytest tests/ -v
```

### Running Python Scripts

Activate the venv before running any Python scripts:

```bash
source .env/bin/activate && python your_script.py
```

### Installing Dependencies

```bash
source .env/bin/activate && pip install -r requirements-dev.txt
```

## Available Commands

- **Run all tests**: `pytest tests/ -v`
- **Run specific test file**: `pytest tests/test_lob.py -v`
- **Install dependencies**: `pip install -r requirements-dev.txt`
- **Format code**: `black .`
- **Lint code**: `ruff check .`
- **Type check**: `mypy .`
