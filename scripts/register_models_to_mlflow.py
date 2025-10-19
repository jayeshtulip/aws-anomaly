"""Register trained models to MLflow Model Registry."""
import sys
import os
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pickle
from loguru import logger
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def register_isolation_forest(
    model_dir: str = "models/isolation_forest",
    model_name: str = "isolation-forest-anomaly-detector"
):
    """Register Isolation Forest model to MLflow.
    
    Args:
        model_dir: Directory containing the model
        model_name: Name for the registered model
    """
    logger.info(f"Registering Isolation Forest model: {model_name}")
    
    # Load the model
    model_path = Path(model_dir) / "model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Start MLflow run
    mlflow.set_experiment("model-registry")
    
    with mlflow.start_run(run_name="register-isolation-forest"):
        # Log the model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Log model metadata
        mlflow.log_param("model_type", "IsolationForest")
        mlflow.log_param("contamination", 0.1)
        
        logger.success(f"✓ Registered {model_name}")


def register_xgboost(
    model_dir: str = "models/xgboost",
    model_name: str = "xgboost-anomaly-classifier"
):
    """Register XGBoost model to MLflow.
    
    Args:
        model_dir: Directory containing the model
        model_name: Name for the registered model
    """
    logger.info(f"Registering XGBoost model: {model_name}")
    
    # Load the model
    model_path = Path(model_dir) / "model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load metrics
    import json
    metrics_path = Path("metrics") / "xgboost_metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Start MLflow run
    mlflow.set_experiment("model-registry")
    
    with mlflow.start_run(run_name="register-xgboost"):
        # Log the model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Log model metadata
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_params({
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100
        })
        
        # Log metrics
        mlflow.log_metrics({
            "test_f1": metrics.get("test_f1", 0),
            "test_roc_auc": metrics.get("test_roc_auc", 0),
            "test_precision": metrics.get("test_precision", 0),
            "test_recall": metrics.get("test_recall", 0)
        })
        
        logger.success(f"✓ Registered {model_name}")


def promote_model_to_production(model_name: str, version: int = 1):
    """Promote a model version to Production stage.
    
    Args:
        model_name: Name of the registered model
        version: Version number to promote
    """
    logger.info(f"Promoting {model_name} version {version} to Production")
    
    client = mlflow.tracking.MlflowClient()
    
    # Transition to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    
    logger.success(f"✓ Model promoted to Production")


def list_registered_models():
    """List all registered models in MLflow."""
    client = mlflow.tracking.MlflowClient()
    
    logger.info("\n=== Registered Models ===")
    
    for rm in client.search_registered_models():
        logger.info(f"\nModel: {rm.name}")
        for mv in rm.latest_versions:
            logger.info(f"  Version {mv.version}: {mv.current_stage}")


if __name__ == "__main__":
    load_dotenv()
    
    # Set MLflow tracking URI
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    
    logger.info("=" * 60)
    logger.info("Registering Models to MLflow Model Registry")
    logger.info("=" * 60)
    
    try:
        # Register models
        register_isolation_forest()
        register_xgboost()
        
        # Promote XGBoost to production
        promote_model_to_production("xgboost-anomaly-classifier", version=1)
        
        # List all registered models
        list_registered_models()
        
        logger.info("\n" + "=" * 60)
        logger.success("✅ Model registration complete!")
        logger.info("=" * 60)
        logger.info("\nView models in MLflow UI: http://localhost:5000")
        
    except Exception as e:
        logger.error(f"❌ Registration failed: {e}")
        import traceback
        traceback.print_exc()