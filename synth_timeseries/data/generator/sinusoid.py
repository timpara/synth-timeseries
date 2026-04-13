import matplotlib.pyplot as plt
import numpy as np


class SinusoidGenerator:
    """
    A class used to generate sinusoidal data.

    ...

    Attributes
    ----------
    n : int
        the number of data points to generate
    n_sinusoid_functions : int
        the number of sinusoidal functions to generate

    Methods
    -------
    f(t, tau, alpha, epsilon):
        Generates a single data point.
    generate_parameters(m):
        Generates the parameters for the sinusoidal functions.
    r(m):
        Generates the sinusoidal data.
    generate_data():
        Generates the data using the sinusoidal functions.
    """

    def __init__(self, n, n_sinusoid_functions):
        """
        Constructs all the necessary attributes for the SinusoidGenerator object.

        Parameters
        ----------
            n : int
                the number of data points to generate
            n_sinusoid_functions : int
                the number of sinusoidal functions to generate

        """
        self.n = n
        self.n_sinusoid_functions = n_sinusoid_functions

    def f(self, t, tau, alpha, epsilon):
        """
        Generates a single data point.

        Parameters
        ----------
            t : int
                the time point
            tau : int
                the period of the sinusoidal function
            alpha : float
                the damping factor of the sinusoidal function
            epsilon : float
                the phase shift of the sinusoidal function

        Returns
        -------
            float
                a single data point
        """
        return np.exp(-alpha * t) * (
            np.sin(3 * np.pi * t / tau + epsilon) / 2
            + np.cos(4 * np.pi * t / tau + epsilon) / 2
        )

    def generate_parameters(self, m):
        """
        Generates the parameters for the sinusoidal functions.
        :param m:
        :return:
        """

        taus = np.random.randint(10, 41, size=m)
        alphas = np.random.uniform(-0.01, 0.01, size=m)
        epsilons = np.random.uniform(-np.pi / 2, np.pi / 2, size=m)
        return np.array([taus, alphas, epsilons]).T

    def r(self, m):
        """
        Generates the sinusoidal data.
        :param m:
        :return:
        """
        params = self.generate_parameters(m)
        t_values = np.arange(self.n)
        result = np.array(
            [[self.f(t, *param_set) for t in t_values] for param_set in params]
        )
        return result

    def generate_data(self):
        """
        Generates the data using the sinusoidal functions.

        Returns
        -------
            numpy.ndarray
                an array of generated data
        """
        return self.r(self.n_sinusoid_functions)


if __name__ == "__main__":
    """
    Example usage of the SinusoidGenerator class.
    """

    # Usage
    generator = SinusoidGenerator(n=50, n_sinusoid_functions=20000)
    data = generator.generate_data()
    plt.plot(data[0])
    plt.show()
