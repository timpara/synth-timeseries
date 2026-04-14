"""Tests for the Ornstein-Uhlenbeck process generator."""

import unittest

import numpy as np

from synth_timeseries.data.generator.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess


class TestOrnsteinUhlenbeckProcess(unittest.TestCase):
    """Test the OrnsteinUhlenbeckProcess class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ou_process = OrnsteinUhlenbeckProcess()

    def test_simulate_returns_ndarray(self):
        """Simulate should return a numpy array."""
        x = self.ou_process.simulate()
        self.assertIsInstance(x, np.ndarray)

    def test_simulate_correct_length(self):
        """Simulate should return array with correct number of steps."""
        x = self.ou_process.simulate()
        self.assertEqual(len(x), self.ou_process.num_steps)

    def test_reset_zeros_state(self):
        """Reset should zero out the internal state."""
        self.ou_process.simulate()
        self.ou_process.reset()
        self.assertTrue(np.all(self.ou_process.x == 0))

    def test_starts_at_zero(self):
        """Process should start at zero by default."""
        x = self.ou_process.simulate()
        self.assertEqual(x[0], 0)

    def test_mean_reversion(self):
        """Over many steps, the process mean should be close to mu."""
        # Use strong mean reversion with many steps
        process = OrnsteinUhlenbeckProcess(
            theta=5.0, mu=10.0, sigma=0.5, dt=0.01, num_steps=10000
        )
        x = process.simulate()
        # The second half should have mean close to mu
        mean_second_half = np.mean(x[5000:])
        self.assertAlmostEqual(mean_second_half, 10.0, delta=2.0)

    def test_custom_parameters(self):
        """Simulate should work with custom OU parameters."""
        process = OrnsteinUhlenbeckProcess(
            theta=0.5, mu=5.0, sigma=0.2, dt=0.001, num_steps=500
        )
        x = process.simulate()
        self.assertEqual(len(x), 500)

    def test_num_steps_one(self):
        """Simulate should handle num_steps=1."""
        process = OrnsteinUhlenbeckProcess(num_steps=1)
        x = process.simulate()
        self.assertEqual(len(x), 1)
        self.assertEqual(x[0], 0)


if __name__ == "__main__":
    unittest.main()
