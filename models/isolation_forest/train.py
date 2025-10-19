"""Train Isolation Forest model with DVC integration."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import pickle
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import sys

# Optional MLflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available")

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

DATA_PATH = Path(params['data_preprocessing']['processed_data_path'])
TEST_SIZE = params['data_preprocessing']['test_size']
RANDOM_STATE = params['data_preprocessing']['random_state']
ISO_PARAMS = params['isolation_forest']
EVAL_PARAMS = params['evaluation']

MODEL_DIR = Path("models/isolation_forest")
METRICS_DIR = Path(EVAL_PARAMS['metrics_output'])
PLOTS_DIR = Path(EVAL_PARAMS['plots_output'])

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load and prepare data."""
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Create anomaly labels
    anomaly_conditions = (
        (df['log_level_ERROR'] == 1) |
        (df['has_error'] == 1) |
        (df['error_rate'] > 0.1) |
        (df['is_slow'] == 1) |
        (df['memory_used_mb'] > df['memory_used_mb'].quantile(0.95)) |
        (df['duration'] > df['duration'].quantile(0.95))
    )
    
    df['is_anomaly'] = anomaly_conditions.astype(int)
    
    # Prepare features
    drop_cols = ['timestamp', 'message', 'log_stream', 'service_name', 'request_id',
                 'error_type', 'error_message', 'raw_json', 'is_anomaly', 'log_level']
    X = df.drop([col for col in drop_cols if col in df.columns], axis=1)
    X = X.select_dtypes(include=[np.number])
    y = df['is_anomaly']
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Anomaly ratio: {y.mean():.2%}")
    
    return X, y


def train_model(X_train, y_train):
    """Train Isolation Forest model."""
    logger.info("Training Isolation Forest model...")
    
    model = IsolationForest(**ISO_PARAMS)
    model.fit(X_train)
    
    logger.info("✓ Model trained successfully")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    logger.info("Evaluating model...")
    
    # Predictions (-1 for anomaly, 1 for normal)
    y_pred_raw = model.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)  # Convert to 0/1
    
    # Decision scores
    scores = model.decision_function(X_test)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'n_samples': len(X_test),
        'n_anomalies': int(y_pred.sum())
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    
    return metrics, y_pred, scores, cm


def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - Isolation Forest', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Confusion matrix saved: {output_path}")


def plot_roc_curve(y_test, scores, output_path):
    """Plot ROC curve."""
    # For Isolation Forest, lower scores mean more anomalous
    # So we need to invert the scores for ROC calculation
    scores_inverted = -scores
    
    try:
        auc = roc_auc_score(y_test, scores_inverted)
        fpr, tpr, _ = roc_curve(y_test, scores_inverted)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Isolation Forest (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Isolation Forest', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ ROC curve saved: {output_path}")
    except Exception as e:
        logger.warning(f"Could not plot ROC curve: {e}")
        # Create empty plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'ROC curve not available', ha='center', va='center')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def save_artifacts(model, scaler, feature_names, metrics):
    """Save model and artifacts."""
    logger.info("Saving model and artifacts...")
    
    # Save model
    with open(MODEL_DIR / "model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open(MODEL_DIR / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open(MODEL_DIR / "feature_names.txt", 'w') as f:
        f.write('\n'.join(feature_names))
    
    # Save metrics for DVC
    with open(METRICS_DIR / 'isolation_forest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✓ Artifacts saved to {MODEL_DIR}")


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("Starting Isolation Forest Training (DVC Pipeline)")
    logger.info("="*60)
    
    # MLflow setup
    mlflow_active = False
    if MLFLOW_AVAILABLE and params.get('mlflow', {}).get('tracking_uri'):
        try:
            mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
            mlflow.set_experiment("isolation-forest-training")
            mlflow.start_run()
            mlflow.log_params(ISO_PARAMS)
            mlflow_active = True
            logger.info("✓ MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Continuing without MLflow.")
    
    try:
        # Load data
        X, y = load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = train_model(X_train_scaled, y_train)
        
        # Evaluate
        metrics, y_pred, scores, cm = evaluate_model(model, X_test_scaled, y_test)
        
        # Generate plots
        if EVAL_PARAMS.get('confusion_matrix'):
            plot_confusion_matrix(cm, PLOTS_DIR / 'isolation_forest_confusion_matrix.png')
        
        if EVAL_PARAMS.get('roc_curve'):
            plot_roc_curve(y_test, scores, PLOTS_DIR / 'isolation_forest_roc_curve.png')
        
        # Save artifacts
        save_artifacts(model, scaler, X.columns.tolist(), metrics)
        
        # MLflow logging
        if mlflow_active:
            try:
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")
                mlflow.log_artifacts(str(PLOTS_DIR))
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")
        
        logger.success("✅ Training complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if mlflow_active:
            try:
                mlflow.end_run()
            except:
                pass


if __name__ == "__main__":
    sys.exit(main())