"""Compare all trained models - DVC integrated."""
import pandas as pd
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import sys

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

DATA_PATH = Path(params['data_preprocessing']['processed_data_path'])
TEST_SIZE = params['data_preprocessing']['test_size']
RANDOM_STATE = params['data_preprocessing']['random_state']
EVAL_PARAMS = params['evaluation']

METRICS_DIR = Path(EVAL_PARAMS['metrics_output'])
PLOTS_DIR = Path(EVAL_PARAMS['plots_output'])

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_path: str, scaler_path: str):
    """Load model and scaler."""
    logger.info(f"Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def evaluate_model(model, scaler, X, y, model_name: str):
    """Evaluate a single model."""
    logger.info(f"Evaluating {model_name}...")
    
    X_scaled = scaler.transform(X)
    y_pred_raw = model.predict(X_scaled)
    
    # Handle Isolation Forest predictions (-1, 1) -> (0, 1)
    if model_name == "Isolation Forest":
        y_pred = (y_pred_raw == -1).astype(int)  # -1 is anomaly -> 1
    else:
        y_pred = y_pred_raw
    
    # Ensure binary classification
    y_pred = np.array(y_pred).astype(int)
    y = np.array(y).astype(int)
    
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_scaled)[:, 1]
        elif hasattr(model, 'decision_function'):
            # For Isolation Forest, invert scores
            scores = model.decision_function(X_scaled)
            y_proba = -scores  # Lower scores = more anomalous
            # Normalize to [0, 1]
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
        else:
            y_proba = None
        
        auc = roc_auc_score(y, y_proba) if y_proba is not None else None
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc = None
    
    metrics = {
        'model': model_name,
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, average='binary', zero_division=0)),
        'recall': float(recall_score(y, y_pred, average='binary', zero_division=0)),
        'f1_score': float(f1_score(y, y_pred, average='binary', zero_division=0)),
        'auc_roc': float(auc) if auc is not None else None
    }
    
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    if auc:
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return metrics


def plot_model_comparison(results, output_path):
    """Create comparison plot."""
    df = pd.DataFrame(results)
    
    # Remove None values
    df = df.fillna(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: All metrics
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        axes[0].bar(x + i*width, df[metric], width, label=metric.replace('_', ' ').title())
    
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width*1.5)
    axes[0].set_xticklabels(df['model'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: F1 Score comparison
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    axes[1].barh(df['model'], df['f1_score'], color=colors[:len(df)])
    axes[1].set_xlabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score by Model', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1.1)
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(df['f1_score']):
        axes[1].text(v + 0.02, i, f'{v:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Comparison plot saved: {output_path}")


def plot_detailed_comparison(results, output_path):
    """Create detailed comparison with heatmap."""
    df = pd.DataFrame(results)
    df = df.fillna(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Heatmap
    metrics_df = df[['accuracy', 'precision', 'recall', 'f1_score']].T
    metrics_df.columns = df['model']
    
    sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title('Metrics Heatmap', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Metric', fontsize=12)
    
    # Plot 2: Grouped bar chart
    metrics_df_T = metrics_df.T
    metrics_df_T.plot(kind='bar', ax=axes[1], width=0.8)
    axes[1].set_title('Metrics by Model', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(title='Metric', loc='lower right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Detailed comparison saved: {output_path}")


def main():
    """Main evaluation pipeline."""
    logger.info("="*60)
    logger.info("Starting Model Evaluation (DVC Pipeline)")
    logger.info("="*60)
    
    try:
        # Load data
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
        
        # Split data (use same split as training)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        logger.info(f"Test set size: {len(X_test)}")
        
        # Evaluate all models
        results = []
        
        models_to_evaluate = [
            ('Isolation Forest', 'models/isolation_forest/model.pkl', 'models/isolation_forest/scaler.pkl'),
            ('XGBoost', 'models/xgboost/model.pkl', 'models/xgboost/scaler.pkl'),
            ('CatBoost', 'models/catboost/model.pkl', 'models/catboost/scaler.pkl')
        ]
        
        for model_name, model_path, scaler_path in models_to_evaluate:
            if Path(model_path).exists() and Path(scaler_path).exists():
                model, scaler = load_model(model_path, scaler_path)
                metrics = evaluate_model(model, scaler, X_test, y_test, model_name)
                results.append(metrics)
            else:
                logger.warning(f"Model not found: {model_path}")
        
        if not results:
            logger.error("No models found to evaluate!")
            return 1
        
        # Save comparison results
        with open(METRICS_DIR / 'model_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Metrics saved: {METRICS_DIR / 'model_comparison.json'}")
        
        # Create comparison plots
        plot_model_comparison(results, PLOTS_DIR / 'model_comparison.png')
        plot_detailed_comparison(results, PLOTS_DIR / 'model_comparison_detailed.png')
        
        # Summary
        df_results = pd.DataFrame(results)
        best_model = df_results.loc[df_results['f1_score'].idxmax()]
        
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"\nBest Model: {best_model['model']}")
        logger.info(f"  F1 Score: {best_model['f1_score']:.4f}")
        logger.info(f"  Accuracy: {best_model['accuracy']:.4f}")
        if best_model['auc_roc']:
            logger.info(f"  AUC-ROC: {best_model['auc_roc']:.4f}")
        logger.info("\n" + "="*60)
        
        logger.success("✅ Evaluation complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())