"""
Custom Prometheus metrics exporter for ML model performance
"""
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time
import random
import mlflow
from typing import Dict

# Define metrics
model_accuracy = Gauge('model_accuracy', 'Current model accuracy', ['model'])
model_f1_score = Gauge('model_f1_score', 'Current F1 score', ['model'])
model_precision = Gauge('model_precision', 'Current precision', ['model'])
model_recall = Gauge('model_recall', 'Current recall', ['model'])

feature_drift_psi = Gauge('feature_drift_psi', 'PSI score for feature drift', ['feature'])
anomaly_rate = Gauge('anomaly_rate', 'Current anomaly detection rate')

model_predictions_total = Counter('model_predictions_total', 'Total predictions', ['model', 'result'])
model_latency_seconds = Histogram('model_latency_seconds', 'Prediction latency', ['model'])

downtime_prevented_minutes = Counter('downtime_prevented_minutes', 'Cumulative downtime prevented')
false_positive_rate = Gauge('false_positive_rate', 'False positive rate')
mttr_minutes = Gauge('mttr_minutes', 'Mean time to resolution')

def update_metrics_from_mlflow():
    """Fetch latest metrics from MLflow"""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get latest XGBoost run
        runs = client.search_runs(
            experiment_ids=["1"],
            filter_string="tags.model_name='xgboost'",
            max_results=1,
            order_by=["start_time DESC"]
        )
        
        if runs:
            run = runs[0]
            metrics = run.data.metrics
            
            model_accuracy.labels(model='xgboost').set(metrics.get('accuracy', 0))
            model_f1_score.labels(model='xgboost').set(metrics.get('f1_score', 0))
            model_precision.labels(model='xgboost').set(metrics.get('precision', 0))
            model_recall.labels(model='xgboost').set(metrics.get('recall', 0))
    
    except Exception as e:
        print(f"Error fetching MLflow metrics: {e}")

def calculate_drift_metrics():
    """Calculate data drift metrics"""
    # This would normally query your production data
    # For now, simulate
    features = ['cpu_mean', 'memory_std', 'error_count', 'network_in', 'cpu_change_rate']
    
    for feature in features:
        psi_score = random.uniform(0.05, 0.35)  # Simulate PSI
        feature_drift_psi.labels(feature=feature).set(psi_score)

def main():
    # Start metrics server
    start_http_server(9100)
    print("Metrics exporter running on port 9100")
    
    while True:
        # Update metrics every 60 seconds
        update_metrics_from_mlflow()
        calculate_drift_metrics()
        
        # Simulate some metrics
        anomaly_rate.set(random.uniform(0.04, 0.06))
        false_positive_rate.set(random.uniform(0.03, 0.05))
        mttr_minutes.set(random.uniform(7, 9))
        
        time.sleep(60)

if __name__ == '__main__':
    main()