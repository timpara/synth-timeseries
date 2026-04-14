"""Ornstein-Uhlenbeck mean-reverting process generator."""

import numpy as np


class OrnsteinUhlenbeckProcess:
    """Ornstein-Uhlenbeck mean-reverting stochastic process.

    The Ornstein-Uhlenbeck (OU) process is a mean-reverting process
    commonly used in finance for modeling interest rates, exchange rates,
    and commodity prices. The process is pulled toward a long-term mean
    with a strength proportional to its distance from that mean.

    The SDE is: dX = theta * (mu - X) * dt + sigma * dW

    Parameters
    ----------
    theta : float, optional
        Speed of mean reversion. Higher values mean faster reversion.
        Default is 0.1.
    mu : float, optional
        Long-term mean (equilibrium level). Default is 0.
    sigma : float, optional
        Volatility parameter. Default is 0.1.
    dt : float, optional
        Time step size. Default is 0.01.
    num_steps : int, optional
        Number of time steps to simulate. Default is 1000.
    """

    def __init__(
        self,
        theta: float = 0.1,
        mu: float = 0,
        sigma: float = 0.1,
        dt: float = 0.01,
        num_steps: int = 1000,
    ) -> None:
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.num_steps = num_steps
        self.reset()

    def reset(self) -> None:
        """Reset the process state to zeros."""
        self.x = np.zeros(self.num_steps)

    def simulate(self) -> np.ndarray:
        """Simulate the Ornstein-Uhlenbeck process.

        Returns
        -------
        numpy.ndarray
            Array of simulated values with length ``num_steps``.
        """
        for i in range(1, self.num_steps):
            dW = np.sqrt(self.dt) * np.random.normal(0, 1)
            dx = self.theta * (self.mu - self.x[i - 1]) * self.dt + self.sigma * dW
            self.x[i] = self.x[i - 1] + dx
        return self.x
