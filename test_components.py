#!/usr/bin/env python3
"""
Simple test script to verify components work individually
"""

def test_imports():
    """Test if all imports work correctly"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit import successful")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imports successful")
    except ImportError as e:
        print(f"❌ Plotly imports failed: {e}")
    
    try:
        from confluent_kafka import Producer
        print("✅ Confluent Kafka import successful")
    except ImportError as e:
        print(f"❌ Confluent Kafka import failed: {e}")
    
    try:
        from pyspark.sql import SparkSession
        print("✅ PySpark import successful")
    except ImportError as e:
        print(f"❌ PySpark import failed: {e}")

def test_dashboard():
    """Test if dashboard can be imported"""
    print("\nTesting dashboard import...")
    try:
        import sys
        sys.path.append('src/dashboard')
        from fraud_monitor import FraudDashboard
        dashboard = FraudDashboard()
        print("✅ Dashboard import and initialization successful")
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")

def test_streaming_detector():
    """Test if streaming detector can be imported"""
    print("\nTesting streaming detector import...")
    try:
        import sys
        sys.path.append('src/streaming')
        from streaming_detector import FraudStreamingDetector
        print("✅ Streaming detector import successful")
    except Exception as e:
        print(f"❌ Streaming detector test failed: {e}")

def test_transaction_simulator():
    """Test if transaction simulator can be imported"""
    print("\nTesting transaction simulator import...")
    try:
        import sys
        sys.path.append('src/streaming')
        from transaction_simulator import TransactionSimulator
        print("✅ Transaction simulator import successful")
    except Exception as e:
        print(f"❌ Transaction simulator test failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing Fraud Detection System Components")
    print("=" * 50)
    
    test_imports()
    test_dashboard()
    test_streaming_detector()
    test_transaction_simulator()
    
    print("\n" + "=" * 50)
    print("✅ Component testing complete!")
    print("\nTo run the full system:")
    print("1. Start Docker: docker compose up -d")
    print("2. Run dashboard only: streamlit run src/dashboard/fraud_monitor.py")
    print("3. Or run full system: ./run_fraud_detection.sh")