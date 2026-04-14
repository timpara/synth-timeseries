"""Sinusoidal time series data generator."""

import numpy as np


class SinusoidGenerator:
    """Generator for synthetic sinusoidal time series data.

    Produces time series by combining damped sinusoidal functions with
    randomized parameters (period, damping, phase shift). Useful for
    generating synthetic periodic signals for testing and benchmarking.

    Parameters
    ----------
    n : int
        Number of time points per series.
    n_sinusoid_functions : int
        Number of sinusoidal series to generate.
    """

    def __init__(self, n: int, n_sinusoid_functions: int) -> None:
        self.n = n
        self.n_sinusoid_functions = n_sinusoid_functions

    def f(self, t: float, tau: float, alpha: float, epsilon: float) -> float:
        """Compute a single damped sinusoidal value.

        Parameters
        ----------
        t : float
            Time point.
        tau : float
            Period of the sinusoidal function.
        alpha : float
            Damping factor (positive values decay, negative values grow).
        epsilon : float
            Phase shift in radians.

        Returns
        -------
        float
            The computed damped sinusoidal value.
        """
        return np.exp(-alpha * t) * (
            np.sin(3 * np.pi * t / tau + epsilon) / 2
            + np.cos(4 * np.pi * t / tau + epsilon) / 2
        )

    def generate_parameters(self, m: int) -> np.ndarray:
        """Generate random parameters for sinusoidal functions.

        Parameters
        ----------
        m : int
            Number of parameter sets to generate.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(m, 3)`` containing ``[tau, alpha, epsilon]``
            for each function.
        """
        taus = np.random.randint(10, 41, size=m)
        alphas = np.random.uniform(-0.01, 0.01, size=m)
        epsilons = np.random.uniform(-np.pi / 2, np.pi / 2, size=m)
        return np.array([taus, alphas, epsilons]).T

    def r(self, m: int) -> np.ndarray:
        """Generate sinusoidal data for ``m`` functions.

        Parameters
        ----------
        m : int
            Number of sinusoidal series to generate.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(m, n)`` with generated sinusoidal data.
        """
        params = self.generate_parameters(m)
        t_values = np.arange(self.n)
        result = np.array(
            [[self.f(t, *param_set) for t in t_values] for param_set in params]
        )
        return result

    def generate_data(self) -> np.ndarray:
        """Generate the full dataset of sinusoidal time series.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_sinusoid_functions, n)`` with generated data.
        """
        return self.r(self.n_sinusoid_functions)
