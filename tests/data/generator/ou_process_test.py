import unittest

import numpy as np

from synth_timeseries.data.generator.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess


class TestOrnsteinUhlenbeckProcess(unittest.TestCase):
    """
    Test the OrnsteinUhlenbeckProcess class
    """

    def setUp(self):
        """
        Set up the test
        :return:
        """
        self.ou_process = OrnsteinUhlenbeckProcess()

    def test_simulate(self):
        """
        Test the simulate method
        :return:
        """
        x = self.ou_process.simulate()
        self.assertEqual(len(x), self.ou_process.num_steps)
        self.assertIsInstance(x, np.ndarray)


if __name__ == "__main__":
    unittest.main()
