"""Simple MLflow test without model registration."""
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

# Set tracking URI
mlflow.set_tracking_uri('http://localhost:5000')

print("Testing MLflow connection...")

# Test connection by listing experiments
try:
    experiments = mlflow.search_experiments()
    print(f"✅ Connected to MLflow!")
    print(f"Found {len(experiments)} experiments")
    
    # Create a simple test run
    mlflow.set_experiment("simple-test")
    
    with mlflow.start_run(run_name="connection-test"):
        mlflow.log_param("test_param", "success")
        mlflow.log_metric("test_metric", 1.0)
        print("✅ Successfully logged to MLflow!")
    
    # Try to list registered models
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()
    
    print(f"\n📦 Registered Models ({len(models)}):")
    for model in models:
        print(f"  - {model.name}")
    
    if len(models) == 0:
        print("  (No models registered yet)")
    
    print("\n✅ All tests passed!")
    print(f"\nView MLflow UI: http://localhost:5000")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()