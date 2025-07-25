import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="Fraud Detection Monitor", layout="wide")


class FraudDashboard:
    def __init__(self):
        st.title("Real-Time Fraud Detection Dashboard")

        # Initialize session state
        if "alerts" not in st.session_state:
            st.session_state.alerts = []
        if "stats" not in st.session_state:
            st.session_state.stats = []

    def load_recent_alerts(self):
        """Load recent fraud alerts"""
        try:
            alerts_df = pd.read_csv(
                "../data/alerts/fraud_log.txt",
                sep=" - ",
                names=["timestamp", "details"],
            )
            return alerts_df.tail(10)
        except:
            return pd.DataFrame()

    def create_metrics_row(self):
        """Display key metrics"""
        col1, col2, col3, col4 = st.columns(4)

        # Simulate real-time metrics (in production, query from Spark)
        with col1:
            st.metric("Transactions/min", "847", "+12%")
        with col2:
            st.metric("Fraud Rate", "2.1%", "+0.3%", delta_color="inverse")
        with col3:
            st.metric("Avg Risk Score", "0.31", "-0.05")
        with col4:
            st.metric("Fraud $ Prevented", "$45,231", "+$5,420")

    def create_real_time_chart(self):
        """Real-time transaction and fraud trend"""
        # Generate sample streaming data
        time_range = pd.date_range(end=datetime.now(), periods=60, freq="1min")

        data = pd.DataFrame(
            {
                "time": time_range,
                "transactions": np.random.poisson(50, 60),
                "frauds": np.random.poisson(1, 60),
            }
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["transactions"],
                name="Total Transactions",
                line=dict(color="blue", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["frauds"],
                name="Fraud Detections",
                line=dict(color="red", width=2),
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="Real-Time Transaction Flow",
            yaxis=dict(title="Transactions"),
            yaxis2=dict(title="Frauds", overlaying="y", side="right"),
            hovermode="x unified",
        )

        return fig

    def create_risk_distribution(self):
        """Risk score distribution"""
        # Sample risk scores
        risk_scores = np.concatenate(
            [
                np.random.beta(2, 5, 1000),  # Normal transactions
                np.random.beta(5, 2, 20),  # Fraud transactions
            ]
        )

        fig = px.histogram(
            risk_scores,
            nbins=50,
            title="Risk Score Distribution",
            labels={"value": "Risk Score", "count": "Frequency"},
        )
        fig.add_vline(
            x=0.7, line_dash="dash", line_color="red", annotation_text="Fraud Threshold"
        )

        return fig

    def create_alert_table(self):
        """Recent fraud alerts table"""
        alerts = pd.DataFrame(
            {
                "Time": pd.date_range(end=datetime.now(), periods=5, freq="5min"),
                "User ID": np.random.randint(1000, 9999, 5),
                "Amount": np.random.uniform(100, 2000, 5),
                "Risk Score": np.random.uniform(0.7, 0.99, 5),
                "Status": ["Blocked", "Under Review", "Blocked", "Flagged", "Blocked"],
            }
        )

        # Color code by status
        def highlight_status(val):
            if val == "Blocked":
                return "background-color: #ff4444"
            elif val == "Under Review":
                return "background-color: #ffaa44"
            else:
                return "background-color: #ffff44"

        styled_alerts = alerts.style.map(highlight_status, subset=["Status"])

        return styled_alerts

    def run(self):
        """Main dashboard loop"""
        # Metrics row
        self.create_metrics_row()

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(self.create_real_time_chart(), use_container_width=True)

        with col2:
            st.plotly_chart(self.create_risk_distribution(), use_container_width=True)

        # Alerts section
        st.subheader("Recent Fraud Alerts")
        st.dataframe(self.create_alert_table(), use_container_width=True)

        # Model performance metrics
        with st.expander("Model Performance Metrics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision", "0.89", "+0.02")
            with col2:
                st.metric("Recall", "0.94", "+0.01")
            with col3:
                st.metric("F1-Score", "0.91", "+0.015")

        # Auto-refresh
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    dashboard = FraudDashboard()
    dashboard.run()
