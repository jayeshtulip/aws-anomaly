"""Hyperparameter optimization using Optuna."""
import pandas as pd
import numpy as np
from pathlib import Path
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
import pickle
import json
from datetime import datetime
from loguru import logger
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Optional MLflow integration
try:
    import mlflow
    import mlflow.xgboost
    from optuna.integration.mlflow import MLflowCallback
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available, skipping MLflow integration")


# Configuration
DATA_PATH = Path("data/processed/features.csv")
STUDY_NAME = "xgboost_anomaly_optimization"
N_TRIALS = 30  # Reduced for faster testing
N_CV_FOLDS = 3  # Reduced for faster testing
RANDOM_SEED = 42
OUTPUT_DIR = Path("ab_testing/optimization_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load and prepare data."""
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    logger.info(f"Original shape: {df.shape}")
    
    # Create anomaly labels based on multiple conditions
    anomaly_conditions = (
        (df['log_level_ERROR'] == 1) |  # Error logs
        (df['has_error'] == 1) |  # Has error flag
        (df['error_rate'] > 0.1) |  # High error rate
        (df['is_slow'] == 1) |  # Slow requests
        (df['memory_used_mb'] > df['memory_used_mb'].quantile(0.95)) |  # High memory
        (df['duration'] > df['duration'].quantile(0.95))  # High duration
    )
    
    df['is_anomaly'] = anomaly_conditions.astype(int)
    
    logger.info(f"Created anomaly labels: {df['is_anomaly'].sum()} anomalies ({df['is_anomaly'].mean():.2%})")
    
    # Drop non-feature columns
    drop_cols = ['timestamp', 'message', 'log_stream', 'service_name', 'request_id', 
                 'error_type', 'error_message', 'raw_json', 'is_anomaly', 'log_level']
    
    X = df.drop([col for col in drop_cols if col in df.columns], axis=1)
    y = df['is_anomaly']
    
    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Number of features: {len(X.columns)}")
    logger.info(f"Numeric features: {X.columns.tolist()}")
    logger.info(f"Anomaly ratio: {y.mean():.2%}")
    
    return X, y


class XGBoostOptimizer:
    """Optimize XGBoost hyperparameters using Optuna."""
    
    def __init__(self, X, y, metric='f1'):
        """Initialize optimizer."""
        self.X = X
        self.y = y
        self.metric = metric
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        # Set up cross-validation
        self.cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        
        # Set up scoring
        if metric == 'f1':
            self.scorer = make_scorer(f1_score)
        elif metric == 'auc':
            self.scorer = make_scorer(roc_auc_score, needs_proba=True)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        logger.info(f"Initialized optimizer with {len(X)} samples, metric: {metric}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Sample hyperparameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Add dart-specific parameters
        if params['booster'] == 'dart':
            params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            params['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 1.0)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 1.0)
        
        # Create model
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(
                model, self.X_scaled, self.y,
                cv=self.cv,
                scoring=self.scorer,
                n_jobs=-1
            )
            
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            # Log to trial
            trial.set_user_attr('mean_score', mean_score)
            trial.set_user_attr('std_score', std_score)
            trial.set_user_attr('cv_scores', cv_scores.tolist())
            
            return mean_score
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
    
    def optimize(self, n_trials: int = N_TRIALS, study_name: str = STUDY_NAME) -> optuna.Study:
        """Run optimization."""
        logger.info(f"Starting optimization with {n_trials} trials...")
    
        # Don't use MLflow callback - we'll log manually
        # This avoids nested run conflicts
    
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
    
        # Optimize (without callbacks)
        study.optimize(
           self.objective,
           n_trials=n_trials,
           show_progress_bar=True
        )
    
        logger.success(f"✅ Optimization complete!")
        logger.info(f"Best {self.metric}: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
    
        return study
    
    
    def train_best_model(self, study: optuna.Study, variant_id: str) -> Dict[str, Any]:
        """Train final model with best parameters."""
        logger.info(f"Training final model: {variant_id}...")
        
        # Get best parameters
        best_params = study.best_params.copy()
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        })
        
        # Train final model
        model = xgb.XGBClassifier(**best_params)
        model.fit(self.X_scaled, self.y)
        
        # Save model
        model_dir = OUTPUT_DIR / variant_id
        model_dir.mkdir(exist_ok=True)
        
        with open(model_dir / "model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        with open(model_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(model_dir / "params.json", 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save feature names
        with open(model_dir / "feature_names.txt", 'w') as f:
            f.write('\n'.join(self.X.columns.tolist()))
        
        logger.info(f"✓ Model saved to {model_dir}")
        
        return {
            'variant_id': variant_id,
            'best_params': best_params,
            'best_score': study.best_value,
            'model_path': str(model_dir / "model.pkl"),
            'scaler_path': str(model_dir / "scaler.pkl")
        }


def analyze_study(study: optuna.Study, output_dir: Path):
    """Analyze and visualize optimization results."""
    logger.info("Analyzing optimization results...")
    
    # Get study dataframe
    df = study.trials_dataframe()
    df.to_csv(output_dir / "trials.csv", index=False)
    
    # Best trial info
    best_trial = study.best_trial
    best_info = {
        'number': best_trial.number,
        'value': best_trial.value,
        'params': best_trial.params,
        'user_attrs': best_trial.user_attrs,
        'datetime_start': best_trial.datetime_start.isoformat(),
        'datetime_complete': best_trial.datetime_complete.isoformat(),
        'duration': (best_trial.datetime_complete - best_trial.datetime_start).total_seconds()
    }
    
    with open(output_dir / "best_trial.json", 'w') as f:
        json.dump(best_info, f, indent=2)
    
    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        with open(output_dir / "param_importance.json", 'w') as f:
            json.dump(importance, f, indent=2)
        
        logger.info("Parameter importance:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {param}: {imp:.4f}")
    except:
        logger.warning("Could not compute parameter importance")
    
    logger.info(f"✓ Analysis complete")
    return best_info


def create_experiment_variants(study: optuna.Study, top_n: int = 3) -> Dict[str, Any]:
    """Create experiment configuration for top N variants."""
    logger.info(f"Creating experiment configuration for top {top_n} variants...")
    
    # Get top trials
    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:top_n]
    
    variants = []
    for i, trial in enumerate(top_trials):
        variant_id = f"xgb_optimized_v{i+1}"
        
        variant = {
            'id': variant_id,
            'name': f"Optimized Variant {i+1}",
            'params': trial.params,
            'score': trial.value,
            'std': trial.user_attrs.get('std_score', 0),
            'weight': 1.0 / top_n
        }
        variants.append(variant)
    
    # Create experiment config
    experiment_config = {
        'name': 'xgboost_optimized_comparison',
        'description': f'Compare top {top_n} optimized XGBoost variants',
        'type': 'hyperparameter',
        'variants': variants,
        'metrics': [
            {'name': 'f1_score', 'threshold': 0.90},
            {'name': 'auc_roc', 'threshold': 0.95},
            {'name': 'latency_p99', 'threshold': 100}
        ],
        'duration_days': 7,
        'min_samples': 1000,
        'created_at': datetime.now().isoformat()
    }
    
    # Save config
    with open(OUTPUT_DIR / "experiment_config.json", 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    logger.info(f"✓ Experiment config saved")
    return experiment_config


def main():
    """Main optimization pipeline."""
    # Set up MLflow if available
    mlflow_active = False
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("xgboost_hyperparameter_optimization")
            mlflow.start_run(run_name=f"optuna_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow_active = True
        except:
            logger.warning("MLflow setup failed, continuing without MLflow")
    
    try:
        # Load data
        X, y = load_data()
        
        logger.info(f"Dataset: {X.shape}, Anomaly ratio: {y.mean():.2%}")
        
        # Initialize optimizer
        optimizer = XGBoostOptimizer(X, y, metric='f1')
        
        # Run optimization
        study = optimizer.optimize(n_trials=N_TRIALS, study_name=STUDY_NAME)
        
        # Analyze results
        best_info = analyze_study(study, OUTPUT_DIR)
        
        # Log to MLflow if available
        if mlflow_active:
            try:
                mlflow.log_params(best_info['params'])
                mlflow.log_metric('best_f1_score', best_info['value'])
                mlflow.log_artifacts(str(OUTPUT_DIR))
            except:
                logger.warning("MLflow logging failed")
        
        # Train best models for top 3 variants
        logger.info("\n" + "="*60)
        logger.info("Training top 3 variants for A/B testing...")
        logger.info("="*60 + "\n")
        
        top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:3]
        
        trained_variants = []
        for i, trial in enumerate(top_trials):
            variant_id = f"xgb_optimized_v{i+1}"
            logger.info(f"\nTraining {variant_id}...")
            logger.info(f"  F1 Score: {trial.value:.4f}")
            logger.info(f"  Params: {trial.params}")
            
            # Train model
            variant_info = optimizer.train_best_model(study, variant_id)
            variant_info['f1_score'] = trial.value
            variant_info['params'] = trial.params
            trained_variants.append(variant_info)
        
        # Create experiment configuration
        experiment_config = create_experiment_variants(study, top_n=3)
        
        # Save summary
        summary = {
            'study_name': STUDY_NAME,
            'n_trials': N_TRIALS,
            'best_score': study.best_value,
            'best_params': study.best_params,
            'top_variants': trained_variants,
            'experiment_config': experiment_config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(OUTPUT_DIR / "optimization_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.success("\n" + "="*60)
        logger.success("✅ OPTIMIZATION COMPLETE!")
        logger.success("="*60)
        logger.info(f"\nResults saved to: {OUTPUT_DIR}")
        logger.info(f"\nBest F1 Score: {study.best_value:.4f}")
        logger.info(f"\nTop 3 variants trained and ready for A/B testing:")
        for variant in trained_variants:
            logger.info(f"  • {variant['variant_id']}: F1={variant['f1_score']:.4f}")
        
        return study, trained_variants, experiment_config
    
    finally:
        if mlflow_active:
            try:
                mlflow.end_run()
            except:
                pass


if __name__ == "__main__":
    study, variants, config = main()