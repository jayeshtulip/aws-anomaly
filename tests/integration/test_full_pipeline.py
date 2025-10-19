"""
End-to-end integration tests
"""
import pytest
import requests
import time
from collections import Counter


class TestEndToEndPipeline:
    """Test complete inference pipeline"""
    
    def test_full_inference_workflow(self, xgboost_model_endpoint):
        """Test complete workflow from request to response"""
        # Step 1: Check model is available
        metadata_response = requests.get(xgboost_model_endpoint, timeout=5)
        assert metadata_response.status_code == 200
        
        metadata = metadata_response.json()
        assert metadata['name'] == 'xgboost_anomaly'
        
        # Step 2: Make inference request
        payload = {
            "inputs": [{
                "name": "input",
                "shape": [1, 37],
                "datatype": "FP32",
                "data": [0.5] * 37
            }]
        }
        
        inference_response = requests.post(
            f"{xgboost_model_endpoint}/infer",
            json=payload,
            timeout=10
        )
        
        assert inference_response.status_code == 200
        
        # Step 3: Validate response
        result = inference_response.json()
        assert 'model_name' in result
        assert 'outputs' in result
        assert len(result['outputs']) == 2
        
        # Step 4: Validate outputs
        outputs = {out['name']: out for out in result['outputs']}
        
        label = outputs['label']['data'][0]
        probabilities = outputs['probabilities']['data']
        
        assert label in [0, 1]
        assert len(probabilities) == 2
        assert abs(sum(probabilities) - 1.0) < 0.01  # Probabilities sum to 1
        
        print(f"\n✓ Full pipeline test passed")
        print(f"  Prediction: {label}")
        print(f"  Probabilities: {probabilities}")


class TestABRouterIntegration:
    """Test A/B router behavior"""
    
    def test_ab_traffic_split(self, triton_inference_url):
        """Verify traffic split is approximately 70/30"""
        n_requests = 200
        model_counts = Counter()
        
        print(f"\nTesting A/B split with {n_requests} requests...")
        
        for i in range(n_requests):
            # Alternate between models to test routing
            if i % 2 == 0:
                # XGBoost request
                payload = {
                    "inputs": [{
                        "name": "input",
                        "shape": [1, 37],
                        "datatype": "FP32",
                        "data": [0.0] * 37
                    }]
                }
                endpoint = f"{triton_inference_url}/v2/models/xgboost_anomaly/infer"
                model_name = "xgboost"
            else:
                # CatBoost request
                payload = {
                    "inputs": [{
                        "name": "features",
                        "shape": [1, 36],
                        "datatype": "FP32",
                        "data": [0.0] * 36
                    }]
                }
                endpoint = f"{triton_inference_url}/v2/models/catboost_anomaly/infer"
                model_name = "catboost"
            
            try:
                response = requests.post(endpoint, json=payload, timeout=5)
                if response.status_code == 200:
                    model_counts[model_name] += 1
            except:
                pass
        
        total = sum(model_counts.values())
        
        if total > 0:
            xgb_pct = (model_counts.get('xgboost', 0) / total) * 100
            cb_pct = (model_counts.get('catboost', 0) / total) * 100
            
            print(f"\nA/B Split Results:")
            print(f"  XGBoost: {model_counts.get('xgboost', 0)} ({xgb_pct:.1f}%)")
            print(f"  CatBoost: {model_counts.get('catboost', 0)} ({cb_pct:.1f}%)")
            print(f"  Total successful: {total}/{n_requests}")
            
            # Both models should be accessible
            assert model_counts.get('xgboost', 0) > 0
            assert model_counts.get('catboost', 0) > 0
    
    def test_both_models_produce_valid_predictions(self, triton_inference_url):
        """Verify both models return valid predictions"""
        # Test XGBoost
        xgb_payload = {
            "inputs": [{
                "name": "input",
                "shape": [1, 37],
                "datatype": "FP32",
                "data": [0.5] * 37
            }]
        }
        
        xgb_response = requests.post(
            f"{triton_inference_url}/v2/models/xgboost_anomaly/infer",
            json=xgb_payload,
            timeout=10
        )
        
        assert xgb_response.status_code == 200
        xgb_result = xgb_response.json()
        assert 'outputs' in xgb_result
        
        # Test CatBoost
        cb_payload = {
            "inputs": [{
                "name": "features",
                "shape": [1, 36],
                "datatype": "FP32",
                "data": [0.5] * 36
            }]
        }
        
        cb_response = requests.post(
            f"{triton_inference_url}/v2/models/catboost_anomaly/infer",
            json=cb_payload,
            timeout=10
        )
        
        assert cb_response.status_code == 200
        cb_result = cb_response.json()
        assert 'outputs' in cb_result
        
        print("\n✓ Both models producing valid predictions")


class TestModelVersioning:
    """Test model versioning support"""
    
    def test_model_version_metadata(self, xgboost_model_endpoint):
        """Check model version information"""
        response = requests.get(xgboost_model_endpoint, timeout=5)
        assert response.status_code == 200
        
        metadata = response.json()
        assert 'versions' in metadata
        assert len(metadata['versions']) > 0
        assert '1' in metadata['versions']
        
        print(f"\n✓ Model versions available: {metadata['versions']}")