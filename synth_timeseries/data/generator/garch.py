import numpy as np


class GARCHProcess:
    """
    The GARCH (Generalized Autoregressive Conditional Heteroskedasticity) process is a statistical model for
    estimating financial market volatility. It models volatility clustering,
    where high volatility periods are often followed by similar periods.
    The GARCHProcess class in Python initializes with parameters defining
    the GARCH model. The reset method initializes arrays for the simulation,
    and the simulate method generates the GARCH process.
    """

    def __init__(self, omega=0.1, alpha=0.1, beta=0.8, dt=0.01, num_steps=1000):
        """
        Initialize the GARCH process.

        Parameters:
            - omega (float): Constant term in the GARCH model. Default: 0.1
            - alpha (float): Coefficient for lagged squared returns in the GARCH model. Default: 0.1
            - beta (float): Coefficient for lagged conditional variances in the GARCH model. Default: 0.8
            - dt (float): Time step size. Default: 0.01
            - num_steps (int): Number of steps for simulation. Default: 1000
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.num_steps = num_steps
        self.reset()

    def reset(self):
        """Reset the process."""
        self.conditional_variance = np.ones(self.num_steps) * self.omega
        self.returns = np.zeros(self.num_steps)

    def simulate(self):
        """Simulate the GARCH process."""
        shocks = np.random.normal(0, 1, size=self.num_steps)
        squared_returns = np.zeros_like(shocks)

        for i in range(1, self.num_steps):
            self.conditional_variance[i] = (
                self.omega
                + self.alpha * squared_returns[i - 1]
                + self.beta * self.conditional_variance[i - 1]
            )
            squared_returns[i] = self.returns[i] ** 2

        self.returns = shocks * np.sqrt(self.conditional_variance)
        return self.returns
