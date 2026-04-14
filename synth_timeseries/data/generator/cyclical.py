"""Seasonal (cyclical) time series data generator with decomposition."""

from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose


class SeasonalDataGenerator:
    """Generator for synthetic seasonal time series data.

    Produces time series with separate seasonal, noise, linear trend,
    cyclic, and three-phase trend components. The components can be
    combined additively or multiplicatively, and decomposed using
    classical seasonal decomposition.

    The three-phase trend cycles through upward, sideways, and downward
    phases, mimicking real-world economic or commodity price patterns.
    """

    def __init__(self) -> None:
        # Constants for the three-phase trend
        self.period = 4 * 30 * 24  # 4 months in hours
        self.upward_slope = 1
        self.downward_slope = -1
        self.sideways_slope = 0
        self.phase_length = self.period // 3

        # Constants for time calculations
        self.day = 24 * 60 * 60
        self.year = 365.2425 * self.day
        self.cycle_period = 5 * self.year

    def generate_three_phase_trend(self, length: int) -> List[float]:
        """Generate a three-phase trend: upward, sideways, downward.

        Parameters
        ----------
        length : int
            Number of data points to generate.

        Returns
        -------
        list of float
            Trend values of the specified length.
        """
        trend: List[float] = []
        if length < self.period:
            for j in range(length):
                trend.append(j * self.upward_slope)
            return trend

        for i in range(length // self.period):
            for j in range(self.phase_length):
                trend.append(j * self.upward_slope + (i * self.period))
            for j in range(self.phase_length):
                trend.append(self.phase_length * self.upward_slope + (i * self.period))
            for j in range(self.phase_length):
                trend.append(
                    self.phase_length * self.upward_slope
                    - j * self.downward_slope
                    + (i * self.period)
                )

        remaining_points = length % self.period
        if remaining_points > 0:
            trend.extend([trend[-1]] * remaining_points)

        return trend[:length]

    def load_combined_components_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with seasonal, noise, trend, and cyclic components.

        Generates hourly data from 2000 to 2023 and combines components
        in both additive and multiplicative ways.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by datetime with columns: ``seasonal``,
            ``noise``, ``linear_trend``, ``cyclic``, ``three_phase_trend``,
            ``multiplicative``, ``additive``.
        """
        df = pd.DataFrame()
        df["date"] = pd.date_range(
            start="2000-01-01 00:00:00", end="2023-12-31 23:59:00", freq="h"
        )

        # Seasonal component (sine wave)
        df["seasonal"] = 1 + np.sin(
            df.date.astype("int64") // 1e9 * (2 * np.pi / self.year)
        )

        # Gaussian noise component
        mean = 0
        std = 0.1
        df["noise"] = np.random.normal(mean, std, len(df))

        # Linear trend component
        linear_slope = 0.0001
        linear_intercept = 100
        df["linear_trend"] = linear_intercept + linear_slope * np.arange(len(df))

        # Cyclic component
        df["cyclic"] = 1 + np.sin(
            df.date.astype("int64") // 1e9 * (2 * np.pi / self.cycle_period)
        )

        # Three-phase trend component
        df["three_phase_trend"] = self.generate_three_phase_trend(len(df))

        # Combine components multiplicatively
        df["multiplicative"] = 1 + df["seasonal"] * (
            df["linear_trend"] + df["three_phase_trend"]
        ) * df["cyclic"] * (1 + df["noise"])

        # Combine components additively
        df["additive"] = (
            df["seasonal"]
            + df["linear_trend"]
            + df["three_phase_trend"]
            + df["cyclic"]
            + df["noise"]
        )

        # Scale and round for clarity
        df["seasonal"] = (df["seasonal"] * 100).round(2)
        df["noise"] = (df["noise"] * 100).round(2)
        df["cyclic"] = (df["cyclic"] * 100).round(2)
        df["linear_trend"] = df["linear_trend"].round(2)
        df["three_phase_trend"] = df["three_phase_trend"].round(2)
        df["multiplicative"] = df["multiplicative"].round(2)
        df["additive"] = df["additive"].round(2)

        df["date"] = df["date"].apply(lambda d: d.strftime("%Y-%m-%d %H:%M:%S"))
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    def generate_seasonal_additive_series(self) -> DecomposeResult:
        """Generate an additive seasonal decomposition.

        Returns
        -------
        statsmodels.tsa.seasonal.DecomposeResult
            Additive decomposition with ``observed``, ``trend``,
            ``seasonal``, and ``resid`` components.
        """
        combined_components_df = self.load_combined_components_dataframe()
        additive_series = combined_components_df["additive"]
        additive_decomposition = seasonal_decompose(
            additive_series, model="additive", period=365 * 24
        )
        return additive_decomposition

    def generate_seasonal_multiplicative_series(self) -> DecomposeResult:
        """Generate a multiplicative seasonal decomposition.

        Returns
        -------
        statsmodels.tsa.seasonal.DecomposeResult
            Multiplicative decomposition with ``observed``, ``trend``,
            ``seasonal``, and ``resid`` components.
        """
        combined_components_df = self.load_combined_components_dataframe()
        multiplicative_series = combined_components_df["multiplicative"]
        multiplicative_decomposition = seasonal_decompose(
            multiplicative_series, model="multiplicative", period=365 * 24
        )
        return multiplicative_decomposition
