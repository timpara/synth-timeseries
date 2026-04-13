import numpy as np


class LevyProcess:
    """
    The Levy process is a type of stochastic process used in mathematical finance.
    It's characterized by stationary independent increments, meaning that the changes in the process
    over non-overlapping intervals are independent and identically distributed. Our `LevyProcess` class in Python
    simulates this process. It initializes with parameters including stability and drift, and uses the Cauchy
    distribution to simulate the process. The `simulate` method generates the Levy process by calculating
    the cumulative sum of the increments.
    """

    def __init__(self, alpha=1.5, beta=0, dt=0.01, num_steps=1000):
        """
        Initialize the Levy process.

        Parameters:
            - alpha (float): Stability parameter. Controls the heavy-tailedness of the process. Default: 1.5
            - beta (float): Drift parameter. Default: 0
            - dt (float): Time step size. Default: 0.01
            - num_steps (int): Number of steps for simulation. Default: 1000
        """
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.num_steps = num_steps
        self.reset()

    def reset(self):
        """Reset the process."""
        self.x = np.zeros(self.num_steps)

    def simulate(self):
        """Simulate the Levy process."""
        dJ = np.random.standard_cauchy(size=self.num_steps) * np.power(
            self.dt, 1 / self.alpha
        )
        dx = self.beta * self.dt + dJ
        self.x = np.cumsum(dx)
        return self.x
