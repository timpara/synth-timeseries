# Contributing to synth-timeseries

Thank you for your interest in contributing! This document provides guidelines
and instructions for contributing to this project.

## Development Setup

1. **Fork and clone** the repository:

   ```bash
   git clone https://github.com/<your-username>/synth-timeseries.git
   cd synth-timeseries
   ```

2. **Install dependencies** using [Poetry](https://python-poetry.org/):

   ```bash
   poetry install
   ```

3. **Install pre-commit hooks**:

   ```bash
   poetry run pre-commit install
   ```

## Running Tests

```bash
poetry run pytest
```

With coverage:

```bash
poetry run pytest --cov=synth_timeseries --cov-report=term-missing
```

## Code Style

This project uses the following tools, enforced via pre-commit hooks:

- **[Black](https://black.readthedocs.io/)** for code formatting
- **[isort](https://pycqa.github.io/isort/)** for import sorting
- **[flake8](https://flake8.pycqa.org/)** for linting
- **[mypy](https://mypy-lang.org/)** for type checking
- **[interrogate](https://interrogate.readthedocs.io/)** for docstring coverage

All public functions and methods must have:

- Type hints on all parameters and return values
- NumPy-style docstrings

## Commit Messages

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new generator for jump-diffusion processes
fix: correct off-by-one in ARIMA differencing
docs: update notebook with GARCH examples
test: add edge case tests for Levy process
refactor: simplify OU process reset logic
```

## Pull Request Process

1. Create a feature branch from `main`:

   ```bash
   git checkout -b feat/your-feature-name
   ```

2. Make your changes, ensuring:
   - All tests pass (`poetry run pytest`)
   - Pre-commit hooks pass (`poetry run pre-commit run --all-files`)
   - New code has type hints and docstrings

3. Push your branch and open a Pull Request against `main`.

4. All CI checks (Lint, Test matrix, Pre-commit) must pass before merging.

## Adding a New Generator

If you are adding a new stochastic process generator:

1. Create the module in `synth_timeseries/data/generator/`
2. Follow the existing pattern: a class with `__init__`, `simulate` (or `generate_data`), and `reset` methods
3. Add the class to `synth_timeseries/data/generator/__init__.py` and `synth_timeseries/__init__.py`
4. Write tests in `tests/data/generator/`
5. Add an example to `notebooks/synthetic_data_generator.ipynb`

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/timpara/synth-timeseries/issues)
to report bugs or request features. Please use the provided issue templates.
