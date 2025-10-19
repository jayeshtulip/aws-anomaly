"""Complete training pipeline: Isolation Forest -> XGBoost -> Evaluation."""
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.isolation_forest.train import train_isolation_forest
from models.xgboost.train import train_xgboost
from scripts.evaluate_models import evaluate_model


def run_training_pipeline():
    """Run complete training pipeline."""
    logger.info("=" * 70)
    logger.info("STARTING COMPLETE TRAINING PIPELINE")
    logger.info("=" * 70)
    
    try:
        # Step 1: Train Isolation Forest
        logger.info("\n[STEP 1/3] Training Isolation Forest...")
        iso_metrics = train_isolation_forest()
        logger.success(f"✓ Isolation Forest trained. Anomaly rate: {iso_metrics['anomaly_rate']:.2%}")
        
        # Step 2: Train XGBoost
        logger.info("\n[STEP 2/3] Training XGBoost...")
        xgb_metrics = train_xgboost()
        logger.success(f"✓ XGBoost trained. Test F1: {xgb_metrics['test_f1']:.4f}")
        
        # Step 3: Evaluate model
        logger.info("\n[STEP 3/3] Evaluating model...")
        eval_metrics = evaluate_model()
        logger.success(f"✓ Evaluation complete. ROC AUC: {eval_metrics['roc_auc']:.4f}")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info("\nFINAL RESULTS:")
        logger.info(f"  Isolation Forest Anomaly Rate: {iso_metrics['anomaly_rate']:.2%}")
        logger.info(f"  XGBoost Test F1 Score:          {xgb_metrics['test_f1']:.4f}")
        logger.info(f"  XGBoost Test ROC AUC:           {xgb_metrics['test_roc_auc']:.4f}")
        logger.info(f"  Final Evaluation ROC AUC:       {eval_metrics['roc_auc']:.4f}")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Set MLflow tracking (local for now)
    os.environ['MLFLOW_TRACKING_URI'] = 'file:./mlruns'
    
    success = run_training_pipeline()
    sys.exit(0 if success else 1)