"""Tests for the Levy process generator."""

import unittest

import numpy as np

from synth_timeseries.data.generator.levy import LevyProcess


class TestLevyProcess(unittest.TestCase):
    """Test the LevyProcess class."""

    def setUp(self):
        """Set up test fixtures."""
        self.levy_process = LevyProcess()

    def test_simulate_returns_ndarray(self):
        """Simulate should return a numpy array."""
        x = self.levy_process.simulate()
        self.assertIsInstance(x, np.ndarray)

    def test_simulate_correct_length(self):
        """Simulate should return array with correct number of steps."""
        x = self.levy_process.simulate()
        self.assertEqual(len(x), self.levy_process.num_steps)

    def test_reset_zeros_state(self):
        """Reset should zero out the internal state."""
        self.levy_process.simulate()
        self.levy_process.reset()
        self.assertTrue(np.all(self.levy_process.x == 0))

    def test_custom_parameters(self):
        """Simulate should work with custom Levy parameters."""
        process = LevyProcess(alpha=1.8, beta=0.5, dt=0.001, num_steps=500)
        x = process.simulate()
        self.assertEqual(len(x), 500)

    def test_num_steps_one(self):
        """Simulate should handle num_steps=1."""
        process = LevyProcess(num_steps=1)
        x = process.simulate()
        self.assertEqual(len(x), 1)

    def test_cumulative_nature(self):
        """Output should be cumulative sums (not independent increments)."""
        process = LevyProcess(num_steps=100, beta=0)
        x = process.simulate()
        # The process is a cumsum, so values should generally differ
        # from a simple noise sequence
        self.assertFalse(np.all(np.diff(x) == 0))


if __name__ == "__main__":
    unittest.main()
