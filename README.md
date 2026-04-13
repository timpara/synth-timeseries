# synth-timeseries

[![Lint and Test](https://github.com/timpara/synth-timeseries/actions/workflows/lint_and_test.yml/badge.svg)](https://github.com/timpara/synth-timeseries/actions/workflows/lint_and_test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A Python library for generating synthetic financial time series using stochastic processes and statistical models.

## Features

This library provides implementations of several widely-used stochastic processes and statistical models for generating realistic synthetic financial time series data:

- **ARIMA** -- Autoregressive Integrated Moving Average processes
- **GARCH** -- Generalized Autoregressive Conditional Heteroskedasticity (volatility modeling)
- **GBM** -- Geometric Brownian Motion (asset price simulation)
- **Levy Process** -- Heavy-tailed stochastic processes
- **Ornstein-Uhlenbeck Process** -- Mean-reverting processes (e.g. interest rates, commodity prices)
- **Cyclical / Seasonal** -- Seasonal decomposition with additive and multiplicative components
- **Sinusoid** -- Parametric sinusoidal data generation

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites

Install Poetry if you haven't already:

```bash
pipx install poetry
```

or

```bash
pip install poetry
```

### Install dependencies

Clone the repository and install:

```bash
git clone https://github.com/timpara/synth-timeseries.git
cd synth-timeseries
poetry install
```

## Quick Start

```python
from synth_timeseries.data.generator.gbm import GeometricBrownianMotion
from synth_timeseries.data.generator.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess
from synth_timeseries.data.generator.garch import GARCHProcess

# Simulate asset prices with Geometric Brownian Motion
gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2, num_steps=252)
prices = gbm.simulate()

# Generate a mean-reverting process
ou = OrnsteinUhlenbeckProcess(theta=0.7, mu=0.0, sigma=0.3, num_steps=252)
rates = ou.simulate()

# Model volatility clustering with GARCH
garch = GARCHProcess(num_steps=252)
returns = garch.simulate()
```

See the `notebooks/` directory for more detailed examples, including seasonal decomposition.

## Testing

Run the test suite with:

```bash
poetry run pytest
```

## Linting

This project uses pre-commit hooks with `black`, `flake8`, `isort`, and `mypy`. Install the hooks with:

```bash
poetry run pre-commit install
```

Run all checks manually:

```bash
poetry run pre-commit run --all-files
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Ensure your code passes tests and linting (`poetry run pytest && poetry run pre-commit run --all-files`)
4. Commit using [conventional commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, etc.)
5. Open a pull request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
