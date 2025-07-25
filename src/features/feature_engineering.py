from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from typing import List


class FraudFeatureEngineer:
    """Production-ready feature engineering for fraud detection"""

    def __init__(self, lookback_windows: List[int] = [5, 10, 30]):
        self.lookback_windows = lookback_windows

    def create_velocity_features(self, df: DataFrame) -> DataFrame:
        """Create transaction velocity features"""
        user_window = Window.partitionBy("user_id").orderBy("Time")

        # Time between transactions
        df = df.withColumn("prev_trans_time", lag("Time", 1).over(user_window))
        df = df.withColumn(
            "time_since_last_trans",
            when(
                col("prev_trans_time").isNotNull(), col("Time") - col("prev_trans_time")
            ).otherwise(999999),
        )

        # Transactions in different time windows
        for hours in [1, 6, 24]:
            window_seconds = hours * 3600
            df = df.withColumn(
                f"trans_in_last_{hours}h",
                sum(
                    when(col("time_since_last_trans") <= window_seconds, 1).otherwise(0)
                ).over(user_window),
            )

        return df

    def create_statistical_features(self, df: DataFrame) -> DataFrame:
        """Create rolling statistical features"""
        for window_size in self.lookback_windows:
            window = (
                Window.partitionBy("user_id")
                .orderBy("Time")
                .rowsBetween(-window_size, -1)
            )

            df = df.withColumn(
                f"mean_amount_last_{window_size}", avg("Amount").over(window)
            )
            df = df.withColumn(
                f"std_amount_last_{window_size}", stddev("Amount").over(window)
            )
            df = df.withColumn(
                f"max_amount_last_{window_size}", max("Amount").over(window)
            )

            # Z-score for current transaction
            df = df.withColumn(
                f"amount_zscore_{window_size}",
                when(
                    col(f"std_amount_last_{window_size}") > 0,
                    (col("Amount") - col(f"mean_amount_last_{window_size}"))
                    / col(f"std_amount_last_{window_size}"),
                ).otherwise(0),
            )

        return df

    def create_pattern_features(self, df: DataFrame) -> DataFrame:
        """Create behavioral pattern features"""
        # Time-based patterns
        df = df.withColumn("hour_of_day", (col("Time") % 86400) / 3600)
        df = df.withColumn(
            "is_weekend", when(col("day_of_week").isin([5, 6]), 1).otherwise(0)
        )
        df = df.withColumn(
            "is_night",
            when((col("hour_of_day") < 6) | (col("hour_of_day") > 22), 1).otherwise(0),
        )

        # Amount patterns
        df = df.withColumn(
            "amount_bin",
            when(col("Amount") < 10, "very_low")
            .when(col("Amount") < 50, "low")
            .when(col("Amount") < 200, "medium")
            .when(col("Amount") < 500, "high")
            .otherwise("very_high"),
        )

        return df

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """Apply all feature engineering"""
        df = self.create_velocity_features(df)
        df = self.create_statistical_features(df)
        df = self.create_pattern_features(df)
        return df
