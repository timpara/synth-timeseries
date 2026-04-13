import unittest

import numpy as np

from synth_timeseries.data.generator.levy import LevyProcess


class TestLevyProcess(unittest.TestCase):
    """
    Test the LevyProcess class
    """

    def setUp(self):
        """
        Set up the test
        :return:
        """
        self.levy_process = LevyProcess()

    def test_simulate(self):
        """
        Test the simulate method
        :return:
        """
        x = self.levy_process.simulate()
        self.assertEqual(len(x), self.levy_process.num_steps)
        self.assertIsInstance(x, np.ndarray)

    def test_reset(self):
        """
        Test the reset method
        :return:
        """
        self.levy_process.simulate()
        self.levy_process.reset()
        self.assertTrue(np.all(self.levy_process.x == 0))


if __name__ == "__main__":
    unittest.main()
