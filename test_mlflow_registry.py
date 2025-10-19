"""Simple MLflow registry test."""
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

# Set tracking URI
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))

# Test connection
print("Testing MLflow connection...")

# Create a test experiment
mlflow.set_experiment("test-registry")

with mlflow.start_run(run_name="test-model-registration"):
    # Log some test parameters
    mlflow.log_param("test_param", "hello")
    mlflow.log_metric("test_metric", 0.95)
    
    # Create a simple sklearn model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10)
    
    # Register it
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="test-model"
    )
    
    print("✅ Model registered successfully!")

# List registered models
client = mlflow.tracking.MlflowClient()
print("\nRegistered models:")
for rm in client.search_registered_models():
    print(f"  - {rm.name}")