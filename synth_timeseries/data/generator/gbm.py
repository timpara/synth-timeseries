"""Geometric Brownian Motion (GBM) process generator."""

import numpy as np


class GeometricBrownianMotion:
    """Geometric Brownian Motion (GBM) process.

    GBM is a continuous-time stochastic process widely used in
    financial modeling for simulating asset prices. It assumes that
    logarithmic returns are normally distributed with constant drift
    and volatility, ensuring that prices remain strictly positive.

    The SDE is: dS = mu * S * dt + sigma * S * dW

    Parameters
    ----------
    mu : float, optional
        Drift parameter (expected return). Default is 0.1.
    sigma : float, optional
        Volatility parameter (standard deviation of returns). Default is 0.2.
    dt : float, optional
        Time step size. Default is 0.01.
    num_steps : int, optional
        Number of time steps to simulate. Default is 1000.
    initial_value : float, optional
        Initial asset price. Default is 1.
    """

    def __init__(
        self,
        mu: float = 0.1,
        sigma: float = 0.2,
        dt: float = 0.01,
        num_steps: int = 1000,
        initial_value: float = 1,
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.num_steps = num_steps
        self.initial_value = initial_value
        self.reset()

    def reset(self) -> None:
        """Reset the process state to zeros."""
        self.x = np.zeros(self.num_steps)

    def simulate(self) -> np.ndarray:
        """Simulate the Geometric Brownian Motion process.

        Returns
        -------
        numpy.ndarray
            Array of simulated asset prices with length ``num_steps``.
            All values are strictly positive.
        """
        dW = np.random.normal(0, np.sqrt(self.dt), size=self.num_steps)
        cumulative_dW = np.cumsum(dW)

        drift = (
            (self.mu - 0.5 * self.sigma**2) * np.arange(1, self.num_steps + 1) * self.dt
        )
        diffusion = self.sigma * cumulative_dW

        self.x = self.initial_value * np.exp(drift + diffusion)
        return self.x
