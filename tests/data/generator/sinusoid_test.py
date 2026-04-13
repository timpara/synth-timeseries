import unittest

import numpy as np

from synth_timeseries.data.generator.sinusoid import SinusoidGenerator


class TestSinusoidGenerator(unittest.TestCase):
    """
    Test the SinusoidGenerator class.
    """

    def setUp(self):
        """
        Set up the test.
        :return:
        """
        self.generator = SinusoidGenerator(n=50, n_sinusoid_functions=20000)

    def test_f(self):
        """
        Test the f method.
        :return:
        """
        result = self.generator.f(1, 20, 0.01, 0)
        self.assertIsInstance(result, float)

    def test_generate_parameters(self):
        """
        Test the generate_parameters method.
        :return:
        """
        result = self.generator.generate_parameters(5)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, 3))

    def test_r(self):
        """
        Test the r method.
        :return:
        """

        result = self.generator.r(5)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, 50))

    def test_generate_data(self):
        """
        Test the generate_data method.
        :return:
        """
        result = self.generator.generate_data()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (20000, 50))


if __name__ == "__main__":
    unittest.main()
