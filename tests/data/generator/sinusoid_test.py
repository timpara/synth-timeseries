"""Tests for the sinusoidal data generator."""

import unittest

import numpy as np

from synth_timeseries.data.generator.sinusoid import SinusoidGenerator


class TestSinusoidGenerator(unittest.TestCase):
    """Test the SinusoidGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = SinusoidGenerator(n=50, n_sinusoid_functions=100)

    def test_f_returns_float(self):
        """The f method should return a float."""
        result = self.generator.f(1, 20, 0.01, 0)
        self.assertIsInstance(result, float)

    def test_f_zero_damping(self):
        """With zero damping, f should return a bounded value."""
        result = self.generator.f(1, 20, 0.0, 0)
        self.assertLessEqual(abs(result), 1.0)

    def test_generate_parameters_shape(self):
        """generate_parameters should return array of shape (m, 3)."""
        result = self.generator.generate_parameters(5)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, 3))

    def test_generate_parameters_ranges(self):
        """Parameter values should be within expected ranges."""
        params = self.generator.generate_parameters(1000)
        taus = params[:, 0]
        alphas = params[:, 1]
        epsilons = params[:, 2]
        self.assertTrue(np.all(taus >= 10))
        self.assertTrue(np.all(taus <= 40))
        self.assertTrue(np.all(alphas >= -0.01))
        self.assertTrue(np.all(alphas <= 0.01))
        self.assertTrue(np.all(epsilons >= -np.pi / 2))
        self.assertTrue(np.all(epsilons <= np.pi / 2))

    def test_r_shape(self):
        """r method should return array of shape (m, n)."""
        result = self.generator.r(5)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, 50))

    def test_generate_data_shape(self):
        """generate_data should return (n_sinusoid_functions, n)."""
        result = self.generator.generate_data()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 50))

    def test_small_n(self):
        """Generator should work with n=1."""
        gen = SinusoidGenerator(n=1, n_sinusoid_functions=3)
        result = gen.generate_data()
        self.assertEqual(result.shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
