# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2024-01-01

### Added

- **Geometric Brownian Motion** (`GeometricBrownianMotion`) -- constant drift and volatility asset price model
- **ARIMA Process** (`ARIMAProcess`) -- autoregressive integrated moving average
- **GARCH Process** (`GARCHProcess`) -- generalized autoregressive conditional heteroskedasticity
- **Ornstein-Uhlenbeck Process** (`OrnsteinUhlenbeckProcess`) -- mean-reverting stochastic process
- **Levy Process** (`LevyProcess`) -- heavy-tailed process with jumps
- **Sinusoidal Generator** (`SinusoidGenerator`) -- damped sinusoidal time series
- **Seasonal Data Generator** (`SeasonalDataGenerator`) -- seasonal decomposition with additive/multiplicative models
- Type hints on all public methods
- NumPy-style docstrings throughout
- Comprehensive test suite (57 tests)
- CI/CD pipelines: Lint, Test (Python 3.9/3.10/3.11), Pre-commit, CodeQL
- Jupyter notebook demonstrating all generators
- PEP 561 `py.typed` marker for type checker support
