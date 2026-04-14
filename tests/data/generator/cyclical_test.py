"""Tests for the seasonal data generator."""

import unittest

import pandas as pd

from synth_timeseries.data.generator.cyclical import SeasonalDataGenerator


class TestSeasonalDataGenerator(unittest.TestCase):
    """Test the SeasonalDataGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = SeasonalDataGenerator()

    def test_generate_three_phase_trend_length(self):
        """Three-phase trend should have the requested length."""
        length = 100
        trend = self.generator.generate_three_phase_trend(length)
        self.assertEqual(len(trend), length)

    def test_generate_three_phase_trend_long(self):
        """Three-phase trend should work for lengths longer than one period."""
        length = self.generator.period * 3 + 500
        trend = self.generator.generate_three_phase_trend(length)
        self.assertEqual(len(trend), length)

    def test_generate_three_phase_trend_short(self):
        """Three-phase trend should handle lengths shorter than one period."""
        length = 10
        trend = self.generator.generate_three_phase_trend(length)
        self.assertEqual(len(trend), length)

    def test_load_combined_components_dataframe(self):
        """DataFrame should contain all expected columns."""
        df = self.generator.load_combined_components_dataframe()
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        expected_columns = [
            "seasonal",
            "noise",
            "linear_trend",
            "cyclic",
            "three_phase_trend",
            "multiplicative",
            "additive",
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_dataframe_index_is_datetime(self):
        """DataFrame index should be a DatetimeIndex."""
        df = self.generator.load_combined_components_dataframe()
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_generate_seasonal_additive_series(self):
        """Additive decomposition should return a decomposition result."""
        result = self.generator.generate_seasonal_additive_series()
        self.assertIsNotNone(result.observed)
        self.assertIsNotNone(result.trend)
        self.assertIsNotNone(result.seasonal)
        self.assertIsNotNone(result.resid)

    def test_generate_seasonal_multiplicative_series(self):
        """Multiplicative decomposition should return a decomposition result."""
        result = self.generator.generate_seasonal_multiplicative_series()
        self.assertIsNotNone(result.observed)
        self.assertIsNotNone(result.trend)
        self.assertIsNotNone(result.seasonal)
        self.assertIsNotNone(result.resid)


if __name__ == "__main__":
    unittest.main()
