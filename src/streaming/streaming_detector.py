from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    sum,
    avg,
    stddev,
    hour,
    dayofweek,
    when,
    window,
    from_json,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    TimestampType,
    DoubleType,
)
from pyspark.ml import PipelineModel
from datetime import datetime


class FraudStreamingDetector:
    def __init__(self, model_path, checkpoint_location):
        self.spark = (
            SparkSession.builder.appName("RealTimeFraudDetection")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.streaming.stateStore.retention", "1h")
            .config("spark.sql.streaming.minBatchesToRetain", "2")
            .getOrCreate()
        )

        self.model = PipelineModel.load(model_path)
        self.checkpoint_location = checkpoint_location

        # Define schema for incoming transactions
        self.transaction_schema = StructType(
            [
                StructField("transaction_id", StringType(), True),
                StructField("user_id", IntegerType(), True),
                StructField("timestamp", TimestampType(), True),
                StructField("amount", DoubleType(), True),
                *[StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)],
            ]
        )

    def create_streaming_features(self, df):
        """Real-time feature engineering with state management"""

        # Add time features
        df = df.withColumn("hour_of_day", hour("timestamp")).withColumn(
            "day_of_week", dayofweek("timestamp")
        )

        # CRITICAL: Streaming aggregations with watermark
        df = df.withWatermark("timestamp", "1 hour")

        # User transaction velocity (streaming window)
        user_velocity = df.groupBy(
            window("timestamp", "1 hour", "5 minutes"), "user_id"
        ).agg(
            count("*").alias("trans_count_1h"),
            sum("amount").alias("sum_amount_1h"),
            avg("amount").alias("avg_amount_1h"),
            stddev("amount").alias("std_amount_1h"),
        )

        # Join back with original stream
        df_with_velocity = df.join(
            user_velocity,
            (df.user_id == user_velocity.user_id)
            & (df.timestamp >= user_velocity.window.start)
            & (df.timestamp <= user_velocity.window.end),
            "left",
        )

        # Add anomaly scores
        df_featured = df_with_velocity.withColumn(
            "amount_zscore",
            when(
                col("std_amount_1h") > 0,
                (col("amount") - col("avg_amount_1h")) / col("std_amount_1h"),
            ).otherwise(0),
        ).withColumn(
            "high_velocity_flag", when(col("trans_count_1h") > 10, 1).otherwise(0)
        )

        return df_featured

    def process_stream(self):
        """Main streaming processing logic"""

        # Read from Kafka
        transaction_stream = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", "localhost:9092")
            .option("subscribe", "transactions")
            .option("startingOffsets", "latest")
            .load()
        )

        # Parse JSON data
        parsed_stream = transaction_stream.select(
            from_json(col("value").cast("string"), self.transaction_schema).alias(
                "data"
            )
        ).select("data.*")

        # Engineer features in real-time
        featured_stream = self.create_streaming_features(parsed_stream)

        # Make predictions
        predictions = self.model.transform(featured_stream)

        # Extract fraud probability
        fraud_alerts = predictions.withColumn(
            "fraud_probability", col("probability").getItem(1)
        ).withColumn("is_fraud", when(col("fraud_probability") > 0.7, 1).otherwise(0))

        # Create alert stream for high-risk transactions
        high_risk = fraud_alerts.filter(col("is_fraud") == 1)

        # Multiple outputs
        # 1. Console output for monitoring
        console_query = (
            high_risk.select(
                "transaction_id", "user_id", "amount", "fraud_probability", "timestamp"
            )
            .writeStream.outputMode("append")
            .format("console")
            .option("truncate", False)
            .trigger(processingTime="10 seconds")
            .start()
        )

        # 2. Write to monitoring database
        def write_to_monitoring(df, epoch_id):
            """Custom sink for alerts"""
            # Convert to Pandas for easier handling
            if not df.isEmpty():
                alerts_pd = df.toPandas()

                # Send to monitoring system (Redis/PostgreSQL)
                for _, alert in alerts_pd.iterrows():
                    # Send alert notification
                    self.send_fraud_alert(alert)

                # Log to file for dashboard
                alerts_pd.to_csv(
                    f"../data/alerts/fraud_alerts_{epoch_id}.csv", index=False
                )

        alert_query = (
            high_risk.writeStream.foreachBatch(write_to_monitoring)
            .outputMode("append")
            .trigger(processingTime="5 seconds")
            .start()
        )

        # 3. Aggregate statistics for dashboard
        stats_query = (
            fraud_alerts.groupBy(window("timestamp", "1 minute"))
            .agg(
                count("*").alias("total_transactions"),
                sum("is_fraud").alias("fraud_count"),
                avg("fraud_probability").alias("avg_risk_score"),
                sum("amount").alias("total_amount"),
                sum(when(col("is_fraud") == 1, col("amount")).otherwise(0)).alias(
                    "fraud_amount"
                ),
            )
            .writeStream.outputMode("complete")
            .format("memory")
            .queryName("fraud_stats")
            .trigger(processingTime="10 seconds")
            .start()
        )

        return console_query, alert_query, stats_query

    def send_fraud_alert(self, alert):
        """Send real-time fraud alert"""
        message = f"""
        FRAUD ALERT
        Transaction ID: {alert["transaction_id"]}
        User ID: {alert["user_id"]}
        Amount: ${alert["amount"]:.2f}
        Fraud Probability: {alert["fraud_probability"]:.2%}
        Time: {alert["timestamp"]}
        """

        # In production: Send to Slack, PagerDuty, email, etc.
        print(message)

        # Log to alert database
        with open("../data/alerts/fraud_log.txt", "a") as f:
            f.write(
                f"{datetime.now()}: {alert['transaction_id']} - Risk: {alert['fraud_probability']:.2%}\n"
            )
