"""
Model inference tests for Triton endpoints
"""
import pytest
import numpy as np
import requests
import time


class TestTritonConnection:
    """Test Triton server connectivity"""
    
    def test_triton_server_reachable(self, triton_inference_url):
        """Test that Triton server is reachable"""
        try:
            response = requests.get(f"{triton_inference_url}/v2/health/live", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Triton server not reachable: {e}")
    
    def test_triton_ready(self, triton_inference_url):
        """Test that Triton server is ready"""
        try:
            response = requests.get(f"{triton_inference_url}/v2/health/ready", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("Triton server not ready")


class TestXGBoostInference:
    """Test XGBoost model inference"""
    
    def test_xgboost_model_available(self, xgboost_model_endpoint):
        """Test that XGBoost model is loaded"""
        try:
            response = requests.get(xgboost_model_endpoint, timeout=5)
            assert response.status_code == 200
            
            data = response.json()
            assert data['name'] == 'xgboost_anomaly'
            assert '1' in data['versions']
        except requests.exceptions.RequestException:
            pytest.skip("XGBoost model not available")
    
    def test_xgboost_inference_valid_input(self, xgboost_model_endpoint, sample_inference_payload):
        """Test XGBoost inference with valid input"""
        try:
            response = requests.post(
                f"{xgboost_model_endpoint}/infer",
                json=sample_inference_payload,
                timeout=10
            )
            
            assert response.status_code == 200
            result = response.json()
            
            assert 'outputs' in result
            assert len(result['outputs']) == 2
            
        except requests.exceptions.RequestException:
            pytest.skip("XGBoost inference endpoint not available")
    
    def test_xgboost_output_format(self, xgboost_model_endpoint, sample_inference_payload):
        """Test XGBoost output format"""
        try:
            response = requests.post(
                f"{xgboost_model_endpoint}/infer",
                json=sample_inference_payload,
                timeout=10
            )
            
            result = response.json()
            outputs = {out['name']: out for out in result['outputs']}
            
            assert 'label' in outputs
            assert outputs['label']['datatype'] == 'INT64'
            assert len(outputs['label']['data']) > 0
            
            assert 'probabilities' in outputs
            assert outputs['probabilities']['datatype'] == 'FP32'
            assert len(outputs['probabilities']['data']) == 2
            
        except requests.exceptions.RequestException:
            pytest.skip("XGBoost inference endpoint not available")
    
    def test_xgboost_inference_latency(self, xgboost_model_endpoint, sample_inference_payload):
        """Test inference latency is within SLA"""
        try:
            start = time.time()
            response = requests.post(
                f"{xgboost_model_endpoint}/infer",
                json=sample_inference_payload,
                timeout=10
            )
            latency = (time.time() - start) * 1000
            
            assert response.status_code == 200
            assert latency < 1000
            
        except requests.exceptions.RequestException:
            pytest.skip("XGBoost inference endpoint not available")


class TestCatBoostInference:
    """Test CatBoost model inference"""
    
    def test_catboost_model_available(self, catboost_model_endpoint):
        """Test that CatBoost model is loaded"""
        try:
            response = requests.get(catboost_model_endpoint, timeout=5)
            assert response.status_code == 200
            
            data = response.json()
            assert data['name'] == 'catboost_anomaly'
            assert '1' in data['versions']
        except requests.exceptions.RequestException:
            pytest.skip("CatBoost model not available")
    
    def test_catboost_inference_valid_input(self, catboost_model_endpoint, sample_inference_payload_catboost):
        """Test CatBoost inference with valid input"""
        try:
            response = requests.post(
                f"{catboost_model_endpoint}/infer",
                json=sample_inference_payload_catboost,
                timeout=10
            )
            
            assert response.status_code == 200
            result = response.json()
            assert 'outputs' in result
            
        except requests.exceptions.RequestException:
            pytest.skip("CatBoost inference endpoint not available")


class TestInferenceEdgeCases:
    """Test edge cases for inference"""
    
    def test_inference_with_zeros(self, xgboost_model_endpoint):
        """Test inference with all zeros"""
        payload = {
            "inputs": [{
                "name": "input",
                "shape": [1, 37],
                "datatype": "FP32",
                "data": [0.0] * 37
            }]
        }
        
        try:
            response = requests.post(
                f"{xgboost_model_endpoint}/infer",
                json=payload,
                timeout=10
            )
            
            assert response.status_code == 200
            
        except requests.exceptions.RequestException:
            pytest.skip("Endpoint not available")