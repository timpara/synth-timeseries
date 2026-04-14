"""Tests for the ARIMA process generator."""

import unittest

import numpy as np

from synth_timeseries.data.generator.arima import ARIMAProcess


class TestARIMAProcess(unittest.TestCase):
    """Test the ARIMAProcess class."""

    def setUp(self):
        """Set up test fixtures."""
        self.arima_process = ARIMAProcess()

    def test_simulate_returns_ndarray(self):
        """Simulate should return a numpy array."""
        x = self.arima_process.simulate()
        self.assertIsInstance(x, np.ndarray)

    def test_simulate_correct_length(self):
        """Simulate should return array with correct number of steps."""
        x = self.arima_process.simulate()
        self.assertEqual(len(x), self.arima_process.num_steps)

    def test_reset_zeros_state(self):
        """Reset should zero out the internal state."""
        self.arima_process.simulate()
        self.arima_process.reset()
        self.assertTrue(np.all(self.arima_process.x == 0))

    def test_custom_parameters(self):
        """Simulate should work with custom AR and MA parameters."""
        process = ARIMAProcess(ar_params=[0.3, 0.2], ma_params=[0.4], num_steps=500)
        x = process.simulate()
        self.assertEqual(len(x), 500)
        self.assertIsInstance(x, np.ndarray)

    def test_differencing(self):
        """Differencing (d > 0) should shorten the output."""
        process = ARIMAProcess(d=1, num_steps=100)
        x = process.simulate()
        self.assertEqual(len(x), 99)

    def test_num_steps_one(self):
        """Simulate should handle num_steps=1."""
        process = ARIMAProcess(num_steps=1)
        x = process.simulate()
        self.assertEqual(len(x), 1)

    def test_mutable_default_isolation(self):
        """Default list parameters should not be shared across instances."""
        p1 = ARIMAProcess()
        p2 = ARIMAProcess()
        # Modifying one instance's params should not affect the other
        p1.ar_params[0] = 999.0
        self.assertNotEqual(p2.ar_params[0], 999.0)

    def test_reproducibility_with_seed(self):
        """Same seed should produce same results."""
        np.random.seed(123)
        x1 = ARIMAProcess(num_steps=50).simulate()
        np.random.seed(123)
        x2 = ARIMAProcess(num_steps=50).simulate()
        np.testing.assert_array_equal(x1, x2)


if __name__ == "__main__":
    unittest.main()
