import numpy as np


class GeometricBrownianMotion:
    """
    The Geometric Brownian Motion (GBM) process is a stochastic process used in financial modeling, particularly for
    asset prices. It assumes that the logarithmic returns of asset prices follow a normal distribution and are
    independent of each other. The GBM process is characterized by a constant drift (mean of the returns) and
    volatility (standard deviation of the returns). In the GeometricBrownianMotion class, the GBM process is
    simulated by generating random samples from a standard normal distribution, computing the cumulative sum
    of these random increments, and combining this with the drift and diffusion terms to compute the GBM path.
    """

    def __init__(self, mu=0.1, sigma=0.2, dt=0.01, num_steps=1000, initial_value=1):
        """
        Initialize the Geometric Brownian Motion process.

        Parameters:
            - mu (float): Drift parameter. Default: 0.1
            - sigma (float): Volatility parameter. Default: 0.2
            - dt (float): Time step size. Default: 0.01
            - num_steps (int): Number of steps for simulation. Default: 1000
            - initial_value (float): Initial value of the process. Default: 1
        """
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.num_steps = num_steps
        self.initial_value = initial_value
        self.reset()

    def reset(self):
        """Reset the process."""
        self.x = np.zeros(self.num_steps)

    def simulate(self):
        """Simulate the Geometric Brownian Motion process."""
        # Generate random samples from a standard normal distribution
        dW = np.random.normal(0, np.sqrt(self.dt), size=self.num_steps)

        # Compute the cumulative sum of random increments
        cumulative_dW = np.cumsum(dW)

        # Compute the drift term
        drift = (
            (self.mu - 0.5 * self.sigma**2)
            * np.arange(1, self.num_steps + 1)
            * self.dt
        )

        # Compute the diffusion term
        diffusion = self.sigma * cumulative_dW

        # Combine drift and diffusion to compute the GBM path
        self.x = self.initial_value * np.exp(drift + diffusion)

        return self.x
