# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %%
from numpy.__config__ import show
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# %%
spark = (
    SparkSession.builder.appName("FraudDetection")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
    .getOrCreate()
)
# %%
df_pandas = pd.read_csv("../data/raw/creditcard.csv")


# %%
print(f"Dataset shape: {df_pandas.shape}")
print(f"Fraud rate: {(df_pandas['Class'].sum() / len(df_pandas)) * 100:.2f}%")
print(f"Total fraudulent transactions: {df_pandas['Class'].sum()}")
print("Done!!")

# %%
df_spark = spark.read.csv("../data/raw/creditcard.csv", header=True, inferSchema=True)
df_spark.printSchema()

fraud_stats = (
    df_spark.groupBy("Class")
    .agg(
        avg("Amount").alias("avg_amount"),
        stddev("Amount").alias("std_amount"),
        min("Amount").alias("min_amount"),
        max("Amount").alias("max_amount"),
        count("*").alias("Count"),
    )
    .show()
)

df_spark.filter(col("Class") == 1).select(
    percentile_approx("Amount", 0.25).alias("Q1"),
    percentile_approx("Amount", 0.50).alias("Median"),
    percentile_approx("Amount", 0.75).alias("Q3"),
).show()


# Analyze fraud vs normal patterns
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Distribution of amounts
fraud = df_pandas[df_pandas["Class"] == 1]
normal = df_pandas[df_pandas["Class"] == 0]

axes[0, 0].hist(
    [normal["Amount"], fraud["Amount"]], bins=30, label=["Normal", "Fraud"], alpha=0.7
)
axes[0, 0].set_title("Transaction Amount Distribution")
axes[0, 0].set_yscale("log")
axes[0, 0].legend()

# Time patterns
axes[0, 1].hist(
    [normal["Time"], fraud["Time"]], bins=48, label=["Normal", "Fraud"], alpha=0.7
)
axes[0, 1].set_title("Transaction Time Distribution")
axes[0, 1].legend()

# Feature correlation with fraud
correlations = df_pandas.corr()["Class"].sort_values(ascending=False)
axes[1, 0].bar(range(len(correlations)), correlations)
axes[1, 0].set_title("Feature Correlation with Fraud")

# Amount by class boxplot
df_pandas.boxplot(column="Amount", by="Class", ax=axes[1, 1])
axes[1, 1].set_title("Amount by Class")
axes[1, 1].set_yscale("log")

plt.tight_layout()
plt.show()
