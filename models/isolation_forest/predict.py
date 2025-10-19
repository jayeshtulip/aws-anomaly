"""Inference with Isolation Forest model."""
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger


class IsolationForestPredictor:
    """Make predictions with trained Isolation Forest."""
    
    def __init__(self, model_dir: str = "models/isolation_forest"):
        """Initialize predictor.
        
        Args:
            model_dir: Directory containing trained model
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        self.load_model()
    
    def load_model(self):
        """Load trained model and scaler."""
        # Load model
        model_path = self.model_dir / "model.pkl"
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded model from {model_path}")
        
        # Load scaler
        scaler_path = self.model_dir / "scaler.pkl"
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {scaler_path}")
        
        # Load feature names
        features_path = self.model_dir / "feature_names.txt"
        with open(features_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f]
        logger.info(f"Loaded {len(self.feature_columns)} feature names")
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Prepared feature array
        """
        # Select required features
        X = df[self.feature_columns].values
        
        # Handle infinite and NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (predictions, anomaly_scores)
        """
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)
        
        # Convert to binary labels (1 = anomaly, 0 = normal)
        labels = (predictions == -1).astype(int)
        
        logger.info(f"Made predictions for {len(X)} samples")
        logger.info(f"Detected {labels.sum()} anomalies ({labels.mean():.2%})")
        
        return labels, anomaly_scores


if __name__ == "__main__":
    # Example usage
    predictor = IsolationForestPredictor()
    
    # Load test data
    df = pd.read_csv("data/processed/features.csv")
    
    # Make predictions
    predictions, scores = predictor.predict(df.head(100))
    
    logger.info(f"Predictions: {predictions[:10]}")
    logger.info(f"Scores: {scores[:10]}")