# Data pipeline

A Python project for the YOLO-VLM anomaly detection pipeline.

## First-time setup

### 1. Install uv

uv installs the Python version and packages this project needs.

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal, then run `uv --version` to check the installation.
See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
for other installation options.

### 2. Install Python and dependencies

```bash
cd data-pipeline
uv python install 3.12
uv sync --locked
```

Use `uv run` before Python commands to use this environment; no manual
activation is needed. Run `uv sync --locked` again after pulling changes that
update dependencies.

## Add dependencies

To add a production package:

```bash
uv add <package-name>
```

To add a development package:

```bash
uv add --dev <package-name>
```

Commit both `pyproject.toml` and `uv.lock` after changing dependencies.

## Linting and formatting

Linting and formatting are executed when save file is executed. To make changes from the terminal:

```bash
uv run ruff check --fix .
uv run ruff format .
```

## Testing

Run the test suite with:

```bash
uv run pytest
```

To run a single test file:

```bash
uv run pytest tests/video_ingestion/test_recording_worker.py
```

To see which parts of the pipeline were exercised by tests:

```bash
uv run pytest --cov=video_ingestion --cov-report=term-missing
```

Tests live in `tests/`, with filenames starting with `test_`.
