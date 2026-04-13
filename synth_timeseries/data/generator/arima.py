import numpy as np


class ARIMAProcess:
    """
    The Autoregressive Integrated Moving Average (ARIMA) process is a type of time series model used in statistics
    and econometrics. It combines autoregressive (AR), differencing (I), and moving average (MA) components.
    The AR part involves modeling the variable of interest using its own lagged values.
    The I part involves differencing the data points to make the time series stationary.
    The MA part involves modeling the error term as a linear combination of error terms occurring
    contemporaneously and at various times in the past. `ARIMAProcess` class:
    """

    def __init__(
        self, ar_params=[0.5], ma_params=[0.5], d=0, mu=0, sigma=1, num_steps=1000
    ):
        """
        Initialize the ARIMA process.

        Parameters:
            - ar_params (list): List of autoregressive parameters. Default: [0.5]
            - ma_params (list): List of moving average parameters. Default: [0.5]
            - d (int): Degree of differencing for the integrated component. Default: 0
            - mu (float): Mean of the process. Default: 0
            - sigma (float): Standard deviation of the process. Default: 1
            - num_steps (int): Number of steps for simulation. Default: 1000
        """
        self.ar_params = np.array(ar_params)
        self.ma_params = np.array(ma_params)
        self.d = d
        self.mu = mu
        self.sigma = sigma
        self.num_steps = num_steps
        self.reset()

    def reset(self):
        """Reset the process."""
        self.x = np.zeros(self.num_steps)

    def simulate(self):
        """Simulate the ARIMA process."""
        # Generate white noise
        white_noise = np.random.normal(self.mu, self.sigma, size=self.num_steps)

        # Generate autoregressive component
        ar_component = np.zeros_like(self.x)
        for i in range(len(self.ar_params)):
            ar_component += self.ar_params[i] * np.roll(self.x, i + 1)

        # Generate moving average component
        ma_component = np.zeros_like(self.x)
        for i in range(len(self.ma_params)):
            ma_component += self.ma_params[i] * np.roll(white_noise, i + 1)

        # Combine components
        self.x = np.cumsum(ar_component - ma_component)

        # Apply differencing
        if self.d > 0:
            self.x = np.diff(self.x, n=self.d)

        return self.x
