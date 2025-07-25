import random
import time
import json
from datetime import datetime, timedelta
import numpy as np
from confluent_kafka import Producer


class TransactionSimulator:
    """Simulates realistic transaction streams with fraud patterns"""

    def __init__(self, fraud_rate=0.02):
        self.fraud_rate = fraud_rate
        self.users = list(range(1, 10001))  # 10k users
        self.producer = Producer({
            'bootstrap.servers': 'localhost:9092',
            'client.id': 'transaction-simulator'
        })

    def generate_normal_transaction(self, user_id):
        """Generate legitimate transaction"""
        # Normal spending patterns
        amount = np.random.lognormal(3.5, 1.5)  # Most transactions $20-200
        amount = min(amount, 1000)  # Cap at $1000

        # PCA features (normal distribution)
        v_features = {f"V{i}": np.random.normal(0, 1) for i in range(1, 29)}

        return {
            "transaction_id": f"T{int(time.time() * 1000)}",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "amount": round(amount, 2),
            **v_features,
        }

    def generate_fraud_transaction(self, user_id):
        """Generate fraudulent transaction with realistic patterns"""
        fraud_pattern = random.choice(["high_amount", "velocity", "unusual_time"])

        if fraud_pattern == "high_amount":
            # Fraudsters often max out cards
            amount = np.random.uniform(500, 5000)
        elif fraud_pattern == "velocity":
            # Multiple quick transactions
            amount = np.random.uniform(50, 200)
        else:
            # Unusual time transactions
            amount = np.random.uniform(100, 500)

        # Fraud patterns in PCA features (shifted distributions)
        v_features = {}
        for i in range(1, 29):
            if i in [1, 2, 3, 4, 5, 9, 10, 11, 12, 14, 16, 17, 18, 19]:
                # Features correlated with fraud (from our analysis)
                v_features[f"V{i}"] = np.random.normal(-2, 1.5)
            else:
                v_features[f"V{i}"] = np.random.normal(0, 1)

        return {
            "transaction_id": f"T{int(time.time() * 1000)}",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "amount": round(amount, 2),
            "fraud_pattern": fraud_pattern,  # For monitoring
            **v_features,
        }

    def simulate_fraud_attack(self, target_user, num_transactions=5):
        """Simulate a fraud attack on specific user"""
        print(f"🚨 Simulating fraud attack on user {target_user}")
        for i in range(num_transactions):
            transaction = self.generate_fraud_transaction(target_user)
            self.producer.produce(
                topic="transactions",
                value=json.dumps(transaction).encode('utf-8')
            )
            self.producer.flush()
            time.sleep(random.uniform(0.5, 2))  # Quick succession

    def run_simulation(self, transactions_per_second=10):
        """Main simulation loop"""
        print("Starting transaction simulation...")
        transaction_count = 0

        while True:
            # Select random user
            user_id = random.choice(self.users)

            # Determine if fraud
            if random.random() < self.fraud_rate:
                # Fraud attack (multiple transactions)
                self.simulate_fraud_attack(user_id)
                transaction_count += 5
            else:
                # Normal transaction
                transaction = self.generate_normal_transaction(user_id)
                self.producer.produce(
                    topic="transactions",
                    value=json.dumps(transaction).encode('utf-8')
                )
                self.producer.flush()
                transaction_count += 1

            # Status update
            if transaction_count % 100 == 0:
                print(f"Generated {transaction_count} transactions...")

            # Control rate
            time.sleep(1 / transactions_per_second)
