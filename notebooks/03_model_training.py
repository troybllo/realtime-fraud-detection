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
#
#

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    RandomForestClassifier,
    GBTClassifier,
    LogisticRegression,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

spark = (
    SparkSession.builder.appName("FraudDetectionModeling")
    .config("spark.driver.memory", "8g")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.host", "localhost")
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
    .getOrCreate()
)

# Load engineered features
df = spark.read.parquet("data/processed/features_engineered.parquet")

# CLASS IMBALANCE

fraud_count = df.filter(F.col("Class") == 1).count()
normal_count = df.filter(F.col("Class") == 0).count()
total_count = df.count()


# calculates balances weights
weight_fraud = total_count / (2 * fraud_count)
weight_normal = total_count / (2 * normal_count)

print(f"Class weights - Fraud: {weight_fraud:.2f}, Normal : {weight_normal:.2f}")

# Add weight column
df = df.withColumn(
    "weight", F.when(F.col("Class") == 1, weight_fraud).otherwise(weight_normal)
)

# Balanced trainign sets with strategies

# Strat A
fraud_df = df.filter(F.col("Class") == 1)
normal_df = df.filter(F.col("Class") == 0)
balanced_df = normal_df.sample(fraction=fraud_count / normal_count, seed=42).union(
    fraud_df
)

# Strategy B: SMOTE (we'll implement a PySpark version)
# Select feature columns
feature_cols = [
    col
    for col in df.columns
    if col not in ["Class", "Time", "user_id", "prev_trans_time", "weight"]
]

# Create feature vector
assembler = VectorAssembler(
    inputCols=feature_cols, outputCol="features_raw", handleInvalid="skip"
)
scaler = StandardScaler(inputCol="features_raw", outputCol="features")

# Split data with stratification
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Ensure fraud cases in both sets
print(
    f"Train fraud rate: {train_df.filter(F.col('Class') == 1).count() / train_df.count():.3%}"
)
print(
    f"Test fraud rate: {test_df.filter(F.col('Class') == 1).count() / test_df.count():.3%}"
)

# Model 1: Logistic Regression (Baseline)
lr = LogisticRegression(
    featuresCol="features",
    labelCol="Class",
    weightCol="weight",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.5,
)

# Model 2 Random Forest (Handles non-linearity)
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="Class",
    weightCol="weight",
    numTrees=100,
    maxDepth=10,
    featureSubsetStrategy="sqrt",
    subsamplingRate=0.8,
)

# Model 3: Gradient Boosted Trees (Best of imbalanced data)
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="Class",
    weightCol="weight",
    maxIter=50,
    maxDepth=5,
    stepSize=0.1,
    subsamplingRate=0.8,
)

# Create pipelines
models = {
    "Logistic Regression": Pipeline(stages=[assembler, scaler, lr]),
    "Random Forest": Pipeline(stages=[assembler, scaler, rf]),
    "Gradient Boosted Trees": Pipeline(stages=[assembler, scaler, gbt]),
}


# Custom evaluation fucntion
def evaluate_fraud_model(model, test_df, model_name):
    predictions = model.transform(test_df)
    pred_pd = predictions.select("Class", "prediction", "probability").toPandas()
    pred_pd["fraud_probability"] = pred_pd["probability"].apply(lambda x: x[1])

    # 1. Base Metrics
    from sklearn.metrics import classification_report, roc_auc_score

    print(f"\n{model_name} Results:")
    print(
        classification_report(
            pred_pd["Class"], pred_pd["prediction"], target_names=["Normal", "Fraud"]
        )
    )

    # 2. ROC-AUC (Critical for imbalanced data)
    roc_auc = roc_auc_score(pred_pd["Class"], pred_pd["fraud_probability"])
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # 3. Precision-Recall Curve (More information than ROC for imbalanced)
    from sklearn.metrics import average_precision_score

    avg_precision = average_precision_score(
        pred_pd["Class"], pred_pd["fraud_probability"]
    )
    print(f"Average Precision Score: {avg_precision:.4f}")

    # 4. Cost-Based Evaluation (Business Impact)
    # Assume: False Negative costs $100, False Positive costs $1
    cm = confusion_matrix(pred_pd["Class"], pred_pd["prediction"])
    tn, fp, fn, tp = cm.ravel()

    cost = (fn * 100) + (fp * 1)
    print(f"Total Cost: ${cost:,}")
    print(f"Fraud Caught: {tp}/{tp + fn} ({tp / (tp + fn):.1%})")

    # 5. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[0, 0])
    axes[0, 0].set_title(f"Confusion Matrix - {model_name}")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")

    # ROC Curve
    fpr, tpr, _ = roc_curve(pred_pd["Class"], pred_pd["fraud_probability"])
    axes[0, 1].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], "k--")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve")
    axes[0, 1].legend()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(
        pred_pd["Class"], pred_pd["fraud_probability"]
    )
    axes[1, 0].plot(recall, precision, label=f"AP = {avg_precision:.3f}")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision-Recall Curve")
    axes[1, 0].legend()

    # Probability Distribution
    axes[1, 1].hist(
        pred_pd[pred_pd["Class"] == 0]["fraud_probability"],
        bins=50,
        alpha=0.5,
        label="Normal",
        density=True,
    )
    axes[1, 1].hist(
        pred_pd[pred_pd["Class"] == 1]["fraud_probability"],
        bins=50,
        alpha=0.5,
        label="Fraud",
        density=True,
    )
    axes[1, 1].set_xlabel("Fraud Probability")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Probability Distribution by Class")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    return predictions, roc_auc


# Train and evaluate all models
results = {}
for name, pipeline in models.items():
    print(f"\nTraining {name}...")
    model = pipeline.fit(train_df)
    predictions, auc = evaluate_fraud_model(model, test_df, name)
    results[name] = {"model": model, "auc": auc, "predictions": predictions}


def optimize_threshold(predictions_df, cost_fn=100, cost_fp=1):
    """Find optimal threshold balancing fraud detection and false positives"""

    pred_pd = predictions_df.select("Class", "probability").toPandas()
    pred_pd["fraud_prob"] = pred_pd["probability"].apply(lambda x: x[1])

    thresholds = np.linspace(0, 1, 100)
    costs = []

    for threshold in thresholds:
        pred_pd["pred"] = (pred_pd["fraud_prob"] >= threshold).astype(int)
        cm = confusion_matrix(pred_pd["Class"], pred_pd["pred"])
        tn, fp, fn, tp = cm.ravel()
        cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(cost)

    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs)
    plt.axvline(
        optimal_threshold,
        color="r",
        linestyle="--",
        label=f"Optimal Threshold: {optimal_threshold:.3f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Total Cost ($)")
    plt.title("Cost vs Threshold")
    plt.legend()
    plt.show()

    return optimal_threshold


# Find optimal threshold for best model
best_model_name = max(results, key=lambda x: results[x]["auc"])
optimal_thresh = optimize_threshold(results[best_model_name]["predictions"])
