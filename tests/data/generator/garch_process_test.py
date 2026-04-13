import unittest

import numpy as np

from synth_timeseries.data.generator.garch import GARCHProcess


class TestGARCHProcess(unittest.TestCase):
    """
    Test the GARCHProcess class
    """

    def setUp(self):
        """
        Set up the test
        :return:
        """
        self.garch_process = GARCHProcess()

    def test_simulate(self):
        """
        Test the simulate method
        :return:
        """
        x = self.garch_process.simulate()
        self.assertEqual(len(x), self.garch_process.num_steps)
        self.assertIsInstance(x, np.ndarray)

    def test_reset(self):
        """
        Test the reset method
        :return:
        """
        self.garch_process.simulate()
        self.garch_process.reset()
        self.assertTrue(np.all(self.garch_process.returns == 0))
        self.assertTrue(
            np.all(self.garch_process.conditional_variance == self.garch_process.omega)
        )


if __name__ == "__main__":
    unittest.main()
