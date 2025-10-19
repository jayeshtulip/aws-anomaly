"""Train XGBoost model with DVC and MLflow integration."""
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
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
    import mlflow.xgboost
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
XGBOOST_PARAMS = params['xgboost']
EVAL_PARAMS = params['evaluation']

MODEL_DIR = Path("models/xgboost")
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
        (df['error_rate'] > params['data_preprocessing']['anomaly_threshold']['error_rate']) |
        (df['is_slow'] == 1) |
        (df['memory_used_mb'] > df['memory_used_mb'].quantile(
            params['data_preprocessing']['anomaly_threshold']['memory_percentile']
        )) |
        (df['duration'] > df['duration'].quantile(
            params['data_preprocessing']['anomaly_threshold']['duration_percentile']
        ))
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


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    logger.info("Training XGBoost model...")
    
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    logger.info("✓ Model trained successfully")
    return model


def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model performance."""
    logger.info("Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'auc_roc': float(roc_auc_score(y_test, y_proba))
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return metrics, y_pred, y_proba, cm


def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - XGBoost', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Confusion matrix saved: {output_path}")


def plot_roc_curve(y_test, y_proba, auc, output_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - XGBoost', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ ROC curve saved: {output_path}")


def plot_feature_importance(model, feature_names, output_path):
    """Plot feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:20]  # Top 20
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 20 Feature Importance - XGBoost', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Feature importance saved: {output_path}")


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
    with open(METRICS_DIR / 'xgboost_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✓ Artifacts saved to {MODEL_DIR}")


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("Starting XGBoost Training (DVC Pipeline)")
    logger.info("="*60)
    
    # MLflow setup
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(params['mlflow']['experiment_name'])
            mlflow.start_run(run_name=f"{params['mlflow']['run_name_prefix']}_xgboost")
            mlflow.log_params(XGBOOST_PARAMS)
        except:
            logger.warning("MLflow tracking failed, continuing without it")
    
    try:
        # Load data
        X, y = load_data()
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(TEST_SIZE + 0.1), random_state=RANDOM_STATE, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Evaluate
        metrics, y_pred, y_proba, cm = evaluate_model(model, X_test_scaled, y_test, scaler)
        
        # Generate plots
        if EVAL_PARAMS.get('confusion_matrix'):
            plot_confusion_matrix(cm, PLOTS_DIR / 'xgboost_confusion_matrix.png')
        
        if EVAL_PARAMS.get('roc_curve'):
            plot_roc_curve(y_test, y_proba, metrics['auc_roc'], PLOTS_DIR / 'xgboost_roc_curve.png')
        
        if EVAL_PARAMS.get('feature_importance'):
            plot_feature_importance(model, X.columns.tolist(), PLOTS_DIR / 'xgboost_feature_importance.png')
        
        # Save artifacts
        save_artifacts(model, scaler, X.columns.tolist(), metrics)
        
        # MLflow logging
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics(metrics)
                mlflow.xgboost.log_model(model, "model")
                mlflow.log_artifacts(str(PLOTS_DIR))
            except:
                pass
        
        logger.success("✅ Training complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    finally:
        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except:
                pass


if __name__ == "__main__":
    sys.exit(main())