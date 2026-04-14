"""Levy process generator."""

import numpy as np


class LevyProcess:
    """Levy process with Cauchy-distributed increments.

    A Levy process is a stochastic process with stationary, independent
    increments. This implementation uses the Cauchy distribution to
    generate heavy-tailed increments, making it suitable for modeling
    financial phenomena with extreme events (fat tails).

    Parameters
    ----------
    alpha : float, optional
        Stability parameter controlling heavy-tailedness.
        Must be in (0, 2]. Default is 1.5.
    beta : float, optional
        Drift parameter. Default is 0.
    dt : float, optional
        Time step size. Default is 0.01.
    num_steps : int, optional
        Number of time steps to simulate. Default is 1000.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        beta: float = 0,
        dt: float = 0.01,
        num_steps: int = 1000,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.num_steps = num_steps
        self.reset()

    def reset(self) -> None:
        """Reset the process state to zeros."""
        self.x = np.zeros(self.num_steps)

    def simulate(self) -> np.ndarray:
        """Simulate the Levy process.

        Returns
        -------
        numpy.ndarray
            Array of simulated cumulative values with length ``num_steps``.
        """
        dJ = np.random.standard_cauchy(size=self.num_steps) * np.power(
            self.dt, 1 / self.alpha
        )
        dx = self.beta * self.dt + dJ
        self.x = np.cumsum(dx)
        return self.x
