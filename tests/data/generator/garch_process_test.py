"""Tests for the GARCH process generator."""

import unittest

import numpy as np

from synth_timeseries.data.generator.garch import GARCHProcess


class TestGARCHProcess(unittest.TestCase):
    """Test the GARCHProcess class."""

    def setUp(self):
        """Set up test fixtures."""
        self.garch_process = GARCHProcess()

    def test_simulate_returns_ndarray(self):
        """Simulate should return a numpy array."""
        x = self.garch_process.simulate()
        self.assertIsInstance(x, np.ndarray)

    def test_simulate_correct_length(self):
        """Simulate should return array with correct number of steps."""
        x = self.garch_process.simulate()
        self.assertEqual(len(x), self.garch_process.num_steps)

    def test_reset_zeros_returns(self):
        """Reset should zero out returns."""
        self.garch_process.simulate()
        self.garch_process.reset()
        self.assertTrue(np.all(self.garch_process.returns == 0))

    def test_reset_resets_variance(self):
        """Reset should reset conditional variance to omega."""
        self.garch_process.simulate()
        self.garch_process.reset()
        self.assertTrue(
            np.all(self.garch_process.conditional_variance == self.garch_process.omega)
        )

    def test_conditional_variance_positive(self):
        """Conditional variance should always be positive."""
        self.garch_process.simulate()
        self.assertTrue(np.all(self.garch_process.conditional_variance > 0))

    def test_custom_parameters(self):
        """Simulate should work with custom GARCH parameters."""
        process = GARCHProcess(omega=0.05, alpha=0.15, beta=0.75, num_steps=500)
        x = process.simulate()
        self.assertEqual(len(x), 500)

    def test_num_steps_one(self):
        """Simulate should handle num_steps=1."""
        process = GARCHProcess(num_steps=1)
        x = process.simulate()
        self.assertEqual(len(x), 1)


if __name__ == "__main__":
    unittest.main()
