import unittest

import numpy as np

from synth_timeseries.data.generator.arima import ARIMAProcess


class TestARIMAProcess(unittest.TestCase):
    """
    Test the ARIMAProcess class
    """

    def setUp(self):
        """
        Set up the test
        :return:
        """
        self.arima_process = ARIMAProcess()

    def test_simulate(self):
        """
        Test the simulate method
        :return:
        """
        x = self.arima_process.simulate()
        self.assertEqual(len(x), self.arima_process.num_steps)
        self.assertIsInstance(x, np.ndarray)

    def test_reset(self):
        """
        Test the reset method
        :return:
        """
        self.arima_process.simulate()
        self.arima_process.reset()
        self.assertTrue(np.all(self.arima_process.x == 0))


if __name__ == "__main__":
    unittest.main()
