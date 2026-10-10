# Data pipeline

Data pipeline scaffold for YOLO-VLM anomaly detection. This directory is a
separate uv project with its own `pyproject.toml`, `uv.lock`, and `.venv`.
Python 3.12 is selected by `.python-version`.

## Install uv

On Windows, run in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

On macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal after installation and check:

```bash
uv --version
```

See the official [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
for alternative installation methods.

## Initialize the environment

Clone this repository if needed, then open a terminal in `data-pipeline`.
If your terminal starts in the repository root, run `cd data-pipeline` first.
All commands below run from the `data-pipeline` directory:

```bash
uv python install 3.12
uv sync --locked
uv run main.py
```

`uv sync --locked` creates the local environment and installs dependencies from
the committed lockfile. `uv run` uses that environment without manual activation.
The current entry point prints `Hello from data-pipeline!`; the project currently
declares no runtime dependencies. Ruff is installed as a development dependency.

### Optional environment activation

You can run commands with `uv run` without activating the environment. To use
`python` directly, activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Run `deactivate` when finished. If PowerShell blocks activation, use
`uv run main.py` or `.\.venv\Scripts\python.exe main.py` instead.

## Add dependencies

Run these commands inside `data-pipeline`, replacing `package-name` with the
package you need:

```bash
uv add package-name
# For development tools:
uv add --dev package-name
```

Commit both `pyproject.toml` and `uv.lock` after changing dependencies so other
contributors can recreate the environment with `uv sync --locked`.
