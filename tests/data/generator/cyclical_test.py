import unittest

from synth_timeseries.data.generator.cyclical import SeasonalDataGenerator


class TestSeasonalDataGenerator(unittest.TestCase):
    """
    Test the SeasonalDataGenerator class
    """

    def setUp(self):
        """
        Set up the test
        :return:
        """

        self.generator = SeasonalDataGenerator()

    def test_generate_three_phase_trend(self):
        """
        Test the generate_three_phase_trend method
        :return:
        """
        length = 100
        trend = self.generator.generate_three_phase_trend(length)
        self.assertEqual(len(trend), length)

    def test_load_combined_components_dataframe(self):
        """
        Test the load_combined_components_dataframe method
        :return:
        """
        df = self.generator.load_combined_components_dataframe()
        self.assertIsNotNone(df)
        self.assertIn("seasonal", df.columns)
        self.assertIn("noise", df.columns)
        self.assertIn("linear_trend", df.columns)
        self.assertIn("cyclic", df.columns)
        self.assertIn("three_phase_trend", df.columns)
        self.assertIn("multiplicative", df.columns)
        self.assertIn("additive", df.columns)


if __name__ == "__main__":
    unittest.main()
