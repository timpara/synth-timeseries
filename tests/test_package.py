"""Tests for top-level package imports."""

import unittest


class TestPackageImports(unittest.TestCase):
    """Test that the public API is accessible from the top-level package."""

    def test_import_version(self):
        """Package should expose __version__."""
        from synth_timeseries import __version__

        self.assertIsInstance(__version__, str)
        self.assertEqual(__version__, "0.1.0")

    def test_import_arima(self):
        """ARIMAProcess should be importable from top-level."""
        from synth_timeseries import ARIMAProcess

        self.assertIsNotNone(ARIMAProcess)

    def test_import_garch(self):
        """GARCHProcess should be importable from top-level."""
        from synth_timeseries import GARCHProcess

        self.assertIsNotNone(GARCHProcess)

    def test_import_gbm(self):
        """GeometricBrownianMotion should be importable from top-level."""
        from synth_timeseries import GeometricBrownianMotion

        self.assertIsNotNone(GeometricBrownianMotion)

    def test_import_levy(self):
        """LevyProcess should be importable from top-level."""
        from synth_timeseries import LevyProcess

        self.assertIsNotNone(LevyProcess)

    def test_import_ou(self):
        """OrnsteinUhlenbeckProcess should be importable from top-level."""
        from synth_timeseries import OrnsteinUhlenbeckProcess

        self.assertIsNotNone(OrnsteinUhlenbeckProcess)

    def test_import_seasonal(self):
        """SeasonalDataGenerator should be importable from top-level."""
        from synth_timeseries import SeasonalDataGenerator

        self.assertIsNotNone(SeasonalDataGenerator)

    def test_import_sinusoid(self):
        """SinusoidGenerator should be importable from top-level."""
        from synth_timeseries import SinusoidGenerator

        self.assertIsNotNone(SinusoidGenerator)


if __name__ == "__main__":
    unittest.main()
