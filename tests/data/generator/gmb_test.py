import unittest

import numpy as np

from synth_timeseries.data.generator.gbm import GeometricBrownianMotion


class TestGBMProcess(unittest.TestCase):
    """
    Test the GeometricBrownianMotion class
    """

    def setUp(self):
        """
        Set up the test
        :return:
        """
        self.gbm_process = GeometricBrownianMotion()

    def test_simulate(self):
        """
        Test the simulate method
        :return:
        """
        x = self.gbm_process.simulate()
        self.assertEqual(len(x), self.gbm_process.num_steps)
        self.assertIsInstance(x, np.ndarray)

    def test_reset(self):
        """
        Test the reset method
        :return:
        """
        self.gbm_process.simulate()
        self.gbm_process.reset()
        self.assertTrue(np.all(self.gbm_process.x == 0))


if __name__ == "__main__":
    unittest.main()
