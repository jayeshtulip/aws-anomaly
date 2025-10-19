"""Quick test of trained model."""
import sys
from pathlib import Path
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from models.xgboost.predict import XGBoostPredictor

# Load predictor
predictor = XGBoostPredictor()

# Load some test data
df = pd.read_csv("data/processed/features.csv").head(10)

# Make predictions
predictions, probabilities = predictor.predict(df)

# Display results
logger.info("\n=== Model Predictions ===")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    status = "ANOMALY" if pred == 1 else "NORMAL"
    logger.info(f"Sample {i+1}: {status} (confidence: {prob:.2%})")