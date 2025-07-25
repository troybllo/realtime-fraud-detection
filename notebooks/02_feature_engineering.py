# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
import pyspark.sql.functions as F

spark = (
    SparkSession.builder.appName("FraudDetectionFeatureEngineering")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.host", "localhost")
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
    .getOrCreate()
)

df = spark.read.csv("data/raw/creditcard.csv", header=True, inferSchema=True)


df = df.withColumn(
    "user_id", F.hash(F.concat(F.col("V1"), F.col("V2"), F.col("V3"))) % 1000
)

df = df.withColumn("hour_of_day", (F.col("Time") % 86400) / 3600).withColumn(
    "day_of_week", F.floor(F.col("Time") / 86400) % 7
)

# 1. VELOCITY FEATURES - How fast are transactions happening?
user_window = Window.partitionBy("user_id").orderBy("Time")
time_window = Window.partitionBy("user_id").orderBy("Time").rowsBetween(-5, -1)

df_features = df.withColumn(
    "prev_trans_time", F.lag("Time", 1).over(user_window)
).withColumn(
    "time_since_last_trans",
    F.when(
        F.col("prev_trans_time").isNotNull(), F.col("Time") - F.col("prev_trans_time")
    ).otherwise(999999),
)

# 2. ROLLING STATISTICS - User's recent behavior
df_features = (
    df_features.withColumn("rolling_mean_amount", F.avg("Amount").over(time_window))
    .withColumn("rolling_std_amount", F.stddev("Amount").over(time_window))
    .withColumn("rolling_count_trans", F.count("*").over(time_window))
)

# 3. ANOMALY SCORES - How unusual is this transaction?
df_features = df_features.withColumn(
    "amount_zscore",
    F.when(
        F.col("rolling_std_amount") > 0,
        (F.col("Amount") - F.col("rolling_mean_amount")) / F.col("rolling_std_amount"),
    ).otherwise(0),
).withColumn(
    "unusual_hour",
    F.when((F.col("hour_of_day") < 6) | (F.col("hour_of_day") > 23), 1).otherwise(0),
)

# 4. TRANSACTION PATTERNS
amount_window = Window.partitionBy("user_id").orderBy("Time").rowsBetween(-10, -1)
df_features = (
    df_features.withColumn(
        "trans_in_last_hour",
        F.sum(F.when(F.col("time_since_last_trans") < 3600, 1).otherwise(0)).over(
            time_window
        ),
    )
    .withColumn("high_amount_flag", F.when(F.col("Amount") > 500, 1).otherwise(0))
    .withColumn("very_low_amount_flag", F.when(F.col("Amount") < 1, 1).otherwise(0))
)

# 5. MERCHANT RISK FEATURES (simulated using V features)
# In real world, you'd have actual merchant data
df_features = df_features.withColumn(
    "merchant_risk_score", F.abs(F.col("V1") + F.col("V2")) / 2
).withColumn("location_risk_score", F.abs(F.col("V3") + F.col("V4")) / 2)

# 6. STATISTICAL OUTLIER DETECTION
# Mahalanobis distance approximation using PCA components
pca_cols = [f"V{i}" for i in range(1, 29)]
pca_squares = [F.col(c) * F.col(c) for c in pca_cols]
pca_sum = pca_squares[0]
for col_expr in pca_squares[1:]:
    pca_sum = pca_sum + col_expr
df_features = df_features.withColumn("pca_magnitude", F.sqrt(pca_sum))

# 7. INTERACTION FEATURES
df_features = df_features.withColumn(
    "amount_velocity_interaction", F.col("Amount") * F.col("trans_in_last_hour")
).withColumn("risk_time_interaction", F.col("unusual_hour") * F.col("high_amount_flag"))

# Show feature engineering results
print("New features created:")
new_features = [
    "time_since_last_trans",
    "rolling_mean_amount",
    "rolling_std_amount",
    "amount_zscore",
    "unusual_hour",
    "trans_in_last_hour",
    "high_amount_flag",
    "merchant_risk_score",
    "pca_magnitude",
]
df_features.select("Class", *new_features).show(100)

# Analyze feature importance for fraud detection
fraud_feature_analysis = (
    df_features.groupBy("Class")
    .agg(*[F.avg(col).alias(f"avg_{col}") for col in new_features])
    .toPandas()
    .T
)

print("Feature differences between fraud and normal:")
print(fraud_feature_analysis)

# Save engineered features
df_features.write.mode("overwrite").parquet(
    "data/processed/features_engineered.parquet"
)
