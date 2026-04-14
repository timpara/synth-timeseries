"""ARIMA (Autoregressive Integrated Moving Average) process generator."""

from typing import List, Optional

import numpy as np


class ARIMAProcess:
    """Autoregressive Integrated Moving Average (ARIMA) process.

    The ARIMA process combines autoregressive (AR), differencing (I),
    and moving average (MA) components to model time series data.
    The AR part models the variable using its own lagged values.
    The I part differences the data to achieve stationarity.
    The MA part models the error term as a linear combination of
    past error terms.

    Parameters
    ----------
    ar_params : list of float, optional
        Autoregressive coefficients. Default is [0.5].
    ma_params : list of float, optional
        Moving average coefficients. Default is [0.5].
    d : int, optional
        Degree of differencing for the integrated component. Default is 0.
    mu : float, optional
        Mean of the white noise process. Default is 0.
    sigma : float, optional
        Standard deviation of the white noise process. Default is 1.
    num_steps : int, optional
        Number of time steps to simulate. Default is 1000.
    """

    def __init__(
        self,
        ar_params: Optional[List[float]] = None,
        ma_params: Optional[List[float]] = None,
        d: int = 0,
        mu: float = 0,
        sigma: float = 1,
        num_steps: int = 1000,
    ) -> None:
        if ar_params is None:
            ar_params = [0.5]
        if ma_params is None:
            ma_params = [0.5]
        self.ar_params = np.array(ar_params)
        self.ma_params = np.array(ma_params)
        self.d = d
        self.mu = mu
        self.sigma = sigma
        self.num_steps = num_steps
        self.reset()

    def reset(self) -> None:
        """Reset the process state to zeros."""
        self.x = np.zeros(self.num_steps)

    def simulate(self) -> np.ndarray:
        """Simulate the ARIMA process.

        Returns
        -------
        numpy.ndarray
            Array of simulated values with length ``num_steps``
            (or shorter if ``d > 0`` due to differencing).
        """
        white_noise = np.random.normal(self.mu, self.sigma, size=self.num_steps)

        ar_component = np.zeros_like(self.x)
        for i in range(len(self.ar_params)):
            ar_component += self.ar_params[i] * np.roll(self.x, i + 1)

        ma_component = np.zeros_like(self.x)
        for i in range(len(self.ma_params)):
            ma_component += self.ma_params[i] * np.roll(white_noise, i + 1)

        self.x = np.cumsum(ar_component - ma_component)

        if self.d > 0:
            self.x = np.diff(self.x, n=self.d)

        return self.x
