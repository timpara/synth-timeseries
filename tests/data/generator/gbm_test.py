"""Tests for the Geometric Brownian Motion process generator."""

import unittest

import numpy as np

from synth_timeseries.data.generator.gbm import GeometricBrownianMotion


class TestGBMProcess(unittest.TestCase):
    """Test the GeometricBrownianMotion class."""

    def setUp(self):
        """Set up test fixtures."""
        self.gbm_process = GeometricBrownianMotion()

    def test_simulate_returns_ndarray(self):
        """Simulate should return a numpy array."""
        x = self.gbm_process.simulate()
        self.assertIsInstance(x, np.ndarray)

    def test_simulate_correct_length(self):
        """Simulate should return array with correct number of steps."""
        x = self.gbm_process.simulate()
        self.assertEqual(len(x), self.gbm_process.num_steps)

    def test_reset_zeros_state(self):
        """Reset should zero out the internal state."""
        self.gbm_process.simulate()
        self.gbm_process.reset()
        self.assertTrue(np.all(self.gbm_process.x == 0))

    def test_all_values_positive(self):
        """GBM prices should always be strictly positive."""
        x = self.gbm_process.simulate()
        self.assertTrue(np.all(x > 0))

    def test_initial_value_respected(self):
        """First value should be close to initial_value."""
        process = GeometricBrownianMotion(initial_value=100, num_steps=1000)
        x = process.simulate()
        # The first value should be exp(small_drift + small_noise) * 100
        # which should be reasonably close to 100
        self.assertAlmostEqual(x[0], 100, delta=10)

    def test_custom_parameters(self):
        """Simulate should work with custom GBM parameters."""
        process = GeometricBrownianMotion(
            mu=0.05, sigma=0.3, dt=0.001, num_steps=500, initial_value=50
        )
        x = process.simulate()
        self.assertEqual(len(x), 500)
        self.assertTrue(np.all(x > 0))

    def test_num_steps_one(self):
        """Simulate should handle num_steps=1."""
        process = GeometricBrownianMotion(num_steps=1)
        x = process.simulate()
        self.assertEqual(len(x), 1)
        self.assertTrue(x[0] > 0)


if __name__ == "__main__":
    unittest.main()
