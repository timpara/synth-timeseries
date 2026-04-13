import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


class SeasonalDataGenerator:
    """
    This class is used to generate the seasonal data for the time series
    """

    def __init__(self):
        """
        This class is used to generate the seasonal data for the time series
        """
        # Constants for the new trend calculations
        self.period = 4 * 30 * 24  # 4 months in hours
        self.upward_slope = 1
        self.downward_slope = -1
        self.sideways_slope = 0
        self.phase_length = (
            self.period // 3
        )  # Divide each period into three phases: upward, sideways, downward

        # Constants for time calculations
        self.day = 24 * 60 * 60
        self.year = 365.2425 * self.day
        self.cycle_period = 5 * self.year  # Example cyclic pattern of 30 days

    def generate_three_phase_trend(self, length):
        """Generate the three-phase trend: upward, sideways, downward."""
        trend = []
        if length < self.period:
            # Upward trend
            for j in range(length):
                trend.append(j * self.upward_slope)
            return trend

        for i in range(length // self.period):
            # Upward trend
            for j in range(self.phase_length):
                trend.append(j * self.upward_slope + (i * self.period))
            # Sideways trend
            for j in range(self.phase_length):
                trend.append(self.phase_length * self.upward_slope + (i * self.period))
            # Downward trend
            for j in range(self.phase_length):
                trend.append(
                    self.phase_length * self.upward_slope
                    - j * self.downward_slope
                    + (i * self.period)
                )

        # Handle any remaining data points outside of complete periods
        remaining_points = length % self.period
        if remaining_points > 0:
            trend.extend(
                [trend[-1]] * remaining_points
            )  # Extend the last value for the remaining points

        # Truncate the trend list to match the length
        return trend[:length]

    def load_combined_components_dataframe(self) -> pd.DataFrame:
        """Create a time series with separate seasonal, Gaussian noise, linear trend, cyclic, and three-phase trend components,
        and combine them in both multiplicative and additive ways."""
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

        # Linear trend component (original one)
        linear_slope = 0.0001  # Adjust the slope as needed
        linear_intercept = 100  # Starting value of the trend component
        df["linear_trend"] = linear_intercept + linear_slope * np.arange(len(df))

        # Cyclic component
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

        # Adjust the scale and round values for clarity
        df["seasonal"] = (df["seasonal"] * 100).round(2)
        df["noise"] = (df["noise"] * 100).round(
            2
        )  # Scaling noise for consistency; adjust as needed
        df["cyclic"] = (df["cyclic"] * 100).round(2)
        df["linear_trend"] = df["linear_trend"].round(2)
        df["three_phase_trend"] = (df["three_phase_trend"]).round(2)  # Adjust as needed
        df["multiplicative"] = df["multiplicative"].round(2)
        df["additive"] = df["additive"].round(2)

        # Convert date to string format for easier viewing
        df["date"] = df["date"].apply(lambda d: d.strftime("%Y-%m-%d %H:%M:%S"))

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    def generate_seasonal_additive_series(self):
        """
        This function generates the additive decomposition of the combined components
        :return:
        """
        # Generate the DataFrame with all combined components
        combined_components_df = self.load_combined_components_dataframe()

        # Additive Decomposition
        additive_series = combined_components_df["additive"]
        additive_decomposition = seasonal_decompose(
            additive_series, model="additive", period=365 * 24
        )  # Use appropriate period
        return additive_decomposition

    def generate_seasonal_multiplicative_series(self):
        """
        This function generates the multiplicative decomposition of the combined components
        :return:
        """
        combined_components_df = self.load_combined_components_dataframe()

        # Multiplicative Decomposition
        multiplicative_series = combined_components_df["multiplicative"]
        multiplicative_decomposition = seasonal_decompose(
            multiplicative_series, model="multiplicative", period=365 * 24
        )  # Use appropriate period
        return multiplicative_decomposition
