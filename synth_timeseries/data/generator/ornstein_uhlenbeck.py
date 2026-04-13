import numpy as np


class OrnsteinUhlenbeckProcess:
    """
    The Ornstein-Uhlenbeck process is a type of stochastic process often used in financial modeling and physics.
    It's a mean-reverting process, meaning it tends to return to its long-term mean value over time.
    This makes it useful for modeling interest rates, exchange rates, and other economic variables that
    tend to revert to a long-term trend
    """

    def __init__(self, theta=0.1, mu=0, sigma=0.1, dt=0.01, num_steps=1000):
        """
        Initialize the Ornstein-Uhlenbeck process.

        Parameters:
            - theta (float): Speed of mean reversion.
            - mu (float): Long-term mean or equilibrium level.
            - sigma (float): Volatility parameter.
            - dt (float): Time step size.
            - num_steps (int): Number of steps for simulation.
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.num_steps = num_steps
        self.reset()

    def reset(self):
        """Reset the process."""
        self.x = np.zeros(self.num_steps)

    def simulate(self):
        """Simulate the Ornstein-Uhlenbeck process."""
        for i in range(1, self.num_steps):
            dW = np.sqrt(self.dt) * np.random.normal(0, 1)
            dx = self.theta * (self.mu - self.x[i - 1]) * self.dt + self.sigma * dW
            self.x[i] = self.x[i - 1] + dx
        return self.x


if __name__ == "__main__":
    ou = OrnsteinUhlenbeckProcess()
    x = ou.simulate()
    print(x)
