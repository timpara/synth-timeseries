"""synth-timeseries: Synthetic financial time series generation.

A Python library for generating synthetic financial time series
using stochastic processes and statistical models.
"""

from synth_timeseries.data.generator.arima import ARIMAProcess
from synth_timeseries.data.generator.cyclical import SeasonalDataGenerator
from synth_timeseries.data.generator.garch import GARCHProcess
from synth_timeseries.data.generator.gbm import GeometricBrownianMotion
from synth_timeseries.data.generator.levy import LevyProcess
from synth_timeseries.data.generator.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess
from synth_timeseries.data.generator.sinusoid import SinusoidGenerator

__version__ = "0.1.0"

__all__ = [
    "ARIMAProcess",
    "GARCHProcess",
    "GeometricBrownianMotion",
    "LevyProcess",
    "OrnsteinUhlenbeckProcess",
    "SeasonalDataGenerator",
    "SinusoidGenerator",
]
