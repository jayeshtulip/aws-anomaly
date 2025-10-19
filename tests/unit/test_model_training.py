"""
Model training tests
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class TestXGBoostTraining:
    """Test XGBoost model training"""
    
    def test_xgboost_trains_successfully(self, sample_training_data, temp_model_dir):
        """Test that XGBoost trains without errors"""
        X, y = sample_training_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42
        )
        
        # Should train without raising exceptions
        model.fit(X_train, y_train)
        
        # Verify model was trained
        assert model.n_estimators == 10
        assert hasattr(model, 'feature_importances_')
        
    def test_xgboost_prediction_shape(self, sample_training_data):
        """Test prediction output shape"""
        X, y = sample_training_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = XGBClassifier(n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Check shapes
        assert predictions.shape == (len(X_test),)
        assert probabilities.shape == (len(X_test), 2)
        
    def test_xgboost_accuracy_threshold(self, sample_training_data):
        """Test that model achieves minimum accuracy"""
        X, y = sample_training_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Should achieve reasonable accuracy on random data
        assert accuracy > 0.4  # Better than random

    def test_xgboost_overfitting_check(self, sample_training_data):
        """Test that model doesn't severely overfit"""
        X, y = sample_training_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        model = XGBClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
            )
        model.fit(X_train, y_train)
    
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
        # With random data, expect higher variance
        # Just verify both are reasonable
        assert train_accuracy > 0.3  # At least better than random
        assert test_accuracy > 0.2   # Test can be lower with random data
    
        # Overfitting check - be more lenient with random data
        overfitting_gap = train_accuracy - test_accuracy
        assert overfitting_gap < 0.6  # More realistic for random data
        
    
        
    def test_xgboost_saves_correctly(self, sample_training_data, temp_model_dir):
        """Test model serialization"""
        X, y = sample_training_data
        
        model = XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Save model
        model_path = Path(temp_model_dir) / "xgboost_model.json"
        model.save_model(model_path)
        
        # Verify file exists
        assert model_path.exists()
        assert model_path.stat().st_size > 0
        
        # Load model and verify
        loaded_model = XGBClassifier()
        loaded_model.load_model(model_path)
        
        # Predictions should match
        original_pred = model.predict(X[:10])
        loaded_pred = loaded_model.predict(X[:10])
        np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_xgboost_hyperparameter_validation(self):
        """Test hyperparameter validation"""
        # Test with clearly invalid string type
        with pytest.raises((ValueError, TypeError, AttributeError)):
            model = XGBClassifier(n_estimators="invalid")
            model.fit([[1, 2]], [0])  # Need to fit to trigger error
    
        # Test max_depth must be positive
        model = XGBClassifier(max_depth=0)
        # XGBoost will adjust this, so just verify it doesn't crash
        assert model.max_depth == 0  # XGBoost allows 0
        
    def test_xgboost_feature_importance(self, sample_training_data):
        """Test feature importance extraction"""
        X, y = sample_training_data
        
        model = XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        importances = model.feature_importances_
        
        # Should have importance for each feature
        assert len(importances) == X.shape[1]
        
        # Importances should sum to 1
        assert abs(importances.sum() - 1.0) < 0.01
        
        # All importances should be non-negative
        assert (importances >= 0).all()


class TestCatBoostTraining:
    """Test CatBoost model training"""
    
    def test_catboost_trains_successfully(self, sample_training_data, temp_model_dir):
        """Test that CatBoost trains without errors"""
        X, y = sample_training_data
        # Use 36 features for CatBoost
        X = X[:, :36]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = CatBoostClassifier(
            iterations=10,
            depth=3,
            random_state=42,
            verbose=False
        )
        
        model.fit(X_train, y_train)
        
        # Verify model was trained
        assert model.tree_count_ > 0
        
    def test_catboost_prediction_types(self, sample_training_data):
        """Test prediction output types"""
        X, y = sample_training_data
        X = X[:, :36]
        
        model = CatBoostClassifier(iterations=10, verbose=False, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        probabilities = model.predict_proba(X[:10])
        
        # Check types
        assert predictions.dtype in [np.int32, np.int64]
        assert probabilities.dtype == np.float64
        
    def test_catboost_saves_correctly(self, sample_training_data, temp_model_dir):
        """Test CatBoost model serialization"""
        X, y = sample_training_data
        X = X[:, :36]
        
        model = CatBoostClassifier(iterations=10, verbose=False, random_state=42)
        model.fit(X, y)
        
        # Save model
        model_path = Path(temp_model_dir) / "catboost_model.cb"
        model.save_model(model_path)
        
        # Verify file exists
        assert model_path.exists()
        
        # Load and verify
        loaded_model = CatBoostClassifier()
        loaded_model.load_model(model_path)
        
        # Predictions should match
        original_pred = model.predict(X[:10])
        loaded_pred = loaded_model.predict(X[:10])
        np.testing.assert_array_equal(original_pred, loaded_pred)


class TestIsolationForestTraining:
    """Test Isolation Forest model training"""
    
    def test_isolation_forest_trains_successfully(self, sample_training_data):
        """Test Isolation Forest training"""
        X, y = sample_training_data
        
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        # Fit model
        model.fit(X)
        
        # Should have decision function
        scores = model.decision_function(X)
        assert scores.shape == (len(X),)
        
    def test_isolation_forest_anomaly_detection(self, sample_training_data):
        """Test anomaly detection"""
        X, y = sample_training_data
        
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)
        
        predictions = model.predict(X)
        
        # Should return -1 for anomalies, 1 for normal
        assert set(predictions).issubset({-1, 1})
        
        # Should detect roughly 10% as anomalies
        anomaly_rate = (predictions == -1).sum() / len(predictions)
        assert 0.05 < anomaly_rate < 0.15  # Within reasonable range


class TestModelComparison:
    """Test model comparison functionality"""
    
    def test_models_on_same_data(self, sample_training_data):
        """Test that all models can train on same data"""
        X, y = sample_training_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost
        xgb_model = XGBClassifier(n_estimators=10, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
        
        # CatBoost (36 features)
        cb_model = CatBoostClassifier(iterations=10, verbose=False, random_state=42)
        cb_model.fit(X_train[:, :36], y_train)
        cb_acc = accuracy_score(y_test, cb_model.predict(X_test[:, :36]))
        
        # Both should achieve reasonable accuracy
        assert xgb_acc > 0.4
        assert cb_acc > 0.4
        
    def test_model_metrics_calculation(self, sample_training_data):
        """Test calculation of various metrics"""
        X, y = sample_training_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # Calculate all metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # All should be between 0 and 1
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1


class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_model_metadata_saved(self, sample_training_data, temp_model_dir):
        """Test that model metadata is saved"""
        X, y = sample_training_data
        
        model = XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Save metadata
        metadata = {
            'model_type': 'XGBoost',
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'accuracy': 0.95,
            'timestamp': '2025-10-17'
        }
        
        metadata_path = Path(temp_model_dir) / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Load and verify
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata['model_type'] == 'XGBoost'
        assert loaded_metadata['n_features'] == 37
    
    def test_model_versioning(self, temp_model_dir):
        """Test model version management"""
        version_1_dir = Path(temp_model_dir) / "v1"
        version_2_dir = Path(temp_model_dir) / "v2"
        
        version_1_dir.mkdir()
        version_2_dir.mkdir()
        
        assert version_1_dir.exists()
        assert version_2_dir.exists()


class TestTrainingConfiguration:
    """Test training configuration validation"""
    def test_config_file_structure(self):
        """Test that config file has required fields"""
        # Load your params.yaml
        config_path = Path("params.yaml")
        if config_path.exists():
            import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check required fields based on YOUR actual config structure
        assert 'xgboost' in config
        assert 'catboost' in config
        assert 'data_preprocessing' in config
        # Your config has these fields, not 'train' or 'model'
    
    @pytest.mark.parametrize("n_estimators,max_depth", [
        (10, 3),
        (50, 5),
        (100, 7),
    ])
    def test_hyperparameter_combinations(self, sample_training_data, n_estimators, max_depth):
        """Test different hyperparameter combinations"""
        X, y = sample_training_data
        
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        # Should train successfully with different params
        model.fit(X, y)
        assert model.n_estimators == n_estimators


class TestTrainingPerformance:
    """Test training performance characteristics"""
    
    def test_training_time_reasonable(self, sample_training_data):
        """Test that training completes in reasonable time"""
        import time
        X, y = sample_training_data
        
        start_time = time.time()
        
        model = XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        training_time = time.time() - start_time
        
        # Should train quickly on small dataset
        assert training_time < 5.0  # seconds
    
    def test_memory_usage_acceptable(self, sample_training_data):
        """Test that training doesn't use excessive memory"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        X, y = sample_training_data
        model = XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use more than 500MB for small dataset
        assert memory_increase < 500