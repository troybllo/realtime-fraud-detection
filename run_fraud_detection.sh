#!/bin/bash

echo "🚀 Starting Real-Time Fraud Detection System"

# Start infrastructure
echo "Starting Kafka..."
docker compose up -d kafka zookeeper

# Wait for Kafka
sleep 10

# Create Kafka topic
docker compose exec kafka kafka-topics --create \
  --topic transactions \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

# Start transaction simulator
echo "Starting transaction simulator..."
python src/streaming/transaction_simulator.py &
SIMULATOR_PID=$!

# Start Spark streaming
echo "Starting Spark streaming detector..."
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 \
  --conf spark.streaming.stopGracefullyOnShutdown=true \
  src/streaming/streaming_detector.py &
SPARK_PID=$!

# Start dashboard
echo "Starting monitoring dashboard..."
streamlit run src/dashboard/fraud_monitor.py &
DASHBOARD_PID=$!

echo "✅ System is running!"
echo "Dashboard: http://localhost:8501"
echo "Press Ctrl+C to stop"

# Wait and cleanup
trap "kill $SIMULATOR_PID $SPARK_PID $DASHBOARD_PID; docker compose down" EXIT
wait
