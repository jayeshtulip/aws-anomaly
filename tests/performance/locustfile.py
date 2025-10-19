"""
Load testing for Triton inference endpoints using Locust
Run with: locust -f tests/performance/locustfile.py --host=http://k8s-triton-tritonab-aafed81da0-1339382983.us-east-1.elb.amazonaws.com
"""
from locust import HttpUser, task, between, events
import random
import json
import time
from typing import Dict, List


class TritonInferenceUser(HttpUser):
    """Simulated user making inference requests"""
    
    wait_time = between(0.1, 0.5)  # Wait 0.1-0.5s between requests
    
    def on_start(self):
        """Initialize user session"""
        # Check if server is healthy
        response = self.client.get("/v2/health/ready")
        if response.status_code != 200:
            print("Warning: Triton server not ready")
    
    def generate_xgboost_payload(self) -> Dict:
        """Generate random payload for XGBoost (37 features)"""
        data = [random.uniform(-3.0, 3.0) for _ in range(37)]
        return {
            "inputs": [{
                "name": "input",
                "shape": [1, 37],
                "datatype": "FP32",
                "data": data
            }]
        }
    
    def generate_catboost_payload(self) -> Dict:
        """Generate random payload for CatBoost (36 features)"""
        data = [random.uniform(-3.0, 3.0) for _ in range(36)]
        return {
            "inputs": [{
                "name": "features",
                "shape": [1, 36],
                "datatype": "FP32",
                "data": data
            }]
        }
    
    @task(7)  # 70% of requests go to XGBoost
    def infer_xgboost(self):
        """Test XGBoost inference endpoint"""
        payload = self.generate_xgboost_payload()
        
        with self.client.post(
            "/v2/models/xgboost_anomaly/infer",
            json=payload,
            catch_response=True,
            name="XGBoost Inference"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if 'outputs' in result and len(result['outputs']) == 2:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)  # 30% of requests go to CatBoost
    def infer_catboost(self):
        """Test CatBoost inference endpoint"""
        payload = self.generate_catboost_payload()
        
        with self.client.post(
            "/v2/models/catboost_anomaly/infer",
            json=payload,
            catch_response=True,
            name="CatBoost Inference"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if 'outputs' in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)  # Occasionally check model metadata
    def get_xgboost_metadata(self):
        """Get XGBoost model metadata"""
        with self.client.get(
            "/v2/models/xgboost_anomaly",
            catch_response=True,
            name="XGBoost Metadata"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)  # Occasionally check health
    def health_check(self):
        """Check server health"""
        with self.client.get(
            "/v2/health/live",
            catch_response=True,
            name="Health Check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class StressTestUser(HttpUser):
    """Stress test with rapid fire requests"""
    
    wait_time = between(0.01, 0.05)  # Very short wait time
    
    def generate_payload(self) -> Dict:
        """Generate random payload"""
        data = [random.uniform(-3.0, 3.0) for _ in range(37)]
        return {
            "inputs": [{
                "name": "input",
                "shape": [1, 37],
                "datatype": "FP32",
                "data": data
            }]
        }
    
    @task
    def rapid_inference(self):
        """Rapid fire inference requests"""
        payload = self.generate_payload()
        
        with self.client.post(
            "/v2/models/xgboost_anomaly/infer",
            json=payload,
            catch_response=True,
            name="Stress Test"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")


# Event handlers for metrics collection
latencies = {
    "xgboost": [],
    "catboost": []
}

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Collect latency data"""
    if exception is None:
        if "XGBoost" in name:
            latencies["xgboost"].append(response_time)
        elif "CatBoost" in name:
            latencies["catboost"].append(response_time)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Calculate and print statistics at end of test"""
    print("\n" + "="*70)
    print("PERFORMANCE TEST RESULTS")
    print("="*70)
    
    for model, times in latencies.items():
        if times:
            times.sort()
            n = len(times)
            
            print(f"\n{model.upper()} Latency Statistics:")
            print(f"  Requests: {n}")
            print(f"  Mean:     {sum(times)/n:.2f}ms")
            print(f"  Median:   {times[n//2]:.2f}ms")
            print(f"  P95:      {times[int(n*0.95)]:.2f}ms")
            print(f"  P99:      {times[int(n*0.99)]:.2f}ms")
            print(f"  Min:      {times[0]:.2f}ms")
            print(f"  Max:      {times[-1]:.2f}ms")
    
    print("\n" + "="*70)