"""
Pytest fixtures for ML testing
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_log_data():
    """Generate sample CloudWatch log data"""
    return {
        'timestamp': '2025-10-17T10:00:00Z',
        'level': 'ERROR',
        'message': 'Database connection timeout',
        'service': 'api-gateway',
        'request_id': 'req-12345',
        'duration_ms': 5000,
        'status_code': 500
    }


@pytest.fixture
def sample_features():
    """Generate sample feature array (37 features for XGBoost)"""
    return np.random.randn(1, 37).astype(np.float32)


@pytest.fixture
def sample_features_catboost():
    """Generate sample feature array (36 features for CatBoost)"""
    return np.random.randn(1, 36).astype(np.float32)


@pytest.fixture
def sample_training_data():
    """Generate sample training dataset"""
    n_samples = 1000
    n_features = 37
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # Binary classification
    
    return X, y


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model artifacts"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def triton_inference_url():
    """Triton inference server URL"""
    return "http://k8s-triton-tritonab-aafed81da0-1339382983.us-east-1.elb.amazonaws.com"


@pytest.fixture
def xgboost_model_endpoint(triton_inference_url):
    """XGBoost model endpoint"""
    return f"{triton_inference_url}/v2/models/xgboost_anomaly"


@pytest.fixture
def catboost_model_endpoint(triton_inference_url):
    """CatBoost model endpoint"""
    return f"{triton_inference_url}/v2/models/catboost_anomaly"


@pytest.fixture
def sample_inference_payload():
    """Sample inference payload for XGBoost"""
    test_data = np.random.randn(37).astype(np.float32).tolist()
    return {
        "inputs": [{
            "name": "input",
            "shape": [1, 37],
            "datatype": "FP32",
            "data": test_data
        }]
    }


@pytest.fixture
def sample_inference_payload_catboost():
    """Sample inference payload for CatBoost"""
    test_data = np.random.randn(36).astype(np.float32).tolist()
    return {
        "inputs": [{
            "name": "features",
            "shape": [1, 36],
            "datatype": "FP32",
            "data": test_data
        }]
    }


@pytest.fixture
def expected_output_schema():
    """Expected output schema from models"""
    return {
        "label": {"type": "INT64", "shape": [1]},
        "probabilities": {"type": "FP32", "shape": [2]}
    }