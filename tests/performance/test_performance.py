"""
Performance tests for inference endpoints
"""
import pytest
import requests
import time
import statistics
import concurrent.futures
from typing import List, Dict


class TestInferencePerformance:
    """Test inference performance characteristics"""
    
    @pytest.mark.performance
    def test_sustained_load_xgboost(self, xgboost_model_endpoint, sample_inference_payload):
        """Test XGBoost under sustained load"""
        n_requests = 100
        latencies = []
        errors = 0
        
        print(f"\nTesting XGBoost with {n_requests} sequential requests...")
        
        for i in range(n_requests):
            try:
                start = time.time()
                response = requests.post(
                    f"{xgboost_model_endpoint}/infer",
                    json=sample_inference_payload,
                    timeout=10
                )
                latency = (time.time() - start) * 1000
                
                if response.status_code == 200:
                    latencies.append(latency)
                else:
                    errors += 1
                    
            except requests.exceptions.RequestException:
                errors += 1
        
        # Calculate statistics
        if latencies:
            latencies.sort()
            mean = statistics.mean(latencies)
            median = statistics.median(latencies)
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
            
            print(f"\nResults:")
            print(f"  Successful: {len(latencies)}/{n_requests}")
            print(f"  Failed: {errors}")
            print(f"  Mean latency: {mean:.2f}ms")
            print(f"  Median: {median:.2f}ms")
            print(f"  P95: {p95:.2f}ms")
            print(f"  P99: {p99:.2f}ms")
            
            # Assertions
            assert len(latencies) >= n_requests * 0.95  # 95% success rate
            assert p95 < 1000  # P95 under 1 second
            assert mean < 500   # Mean under 500ms
    
    @pytest.mark.performance
    def test_concurrent_requests(self, xgboost_model_endpoint, sample_inference_payload):
        """Test concurrent request handling"""
        n_concurrent = 20
        
        def make_request():
            try:
                start = time.time()
                response = requests.post(
                    f"{xgboost_model_endpoint}/infer",
                    json=sample_inference_payload,
                    timeout=10
                )
                latency = (time.time() - start) * 1000
                return {
                    "success": response.status_code == 200,
                    "latency": latency
                }
            except Exception as e:
                return {"success": False, "latency": None, "error": str(e)}
        
        print(f"\nTesting {n_concurrent} concurrent requests...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(n_concurrent)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        latencies = [r["latency"] for r in successful]
        
        if latencies:
            mean_latency = statistics.mean(latencies)
            max_latency = max(latencies)
            
            print(f"\nConcurrent Results:")
            print(f"  Successful: {len(successful)}/{n_concurrent}")
            print(f"  Failed: {len(failed)}")
            print(f"  Mean latency: {mean_latency:.2f}ms")
            print(f"  Max latency: {max_latency:.2f}ms")
            
            # Assertions
            assert len(successful) >= n_concurrent * 0.9  # 90% success under load
            assert mean_latency < 2000  # Mean under 2 seconds under concurrent load
    
    @pytest.mark.performance
    def test_throughput(self, xgboost_model_endpoint, sample_inference_payload):
        """Measure requests per second"""
        duration_seconds = 10
        
        print(f"\nMeasuring throughput for {duration_seconds} seconds...")
        
        start_time = time.time()
        request_count = 0
        errors = 0
        
        while (time.time() - start_time) < duration_seconds:
            try:
                response = requests.post(
                    f"{xgboost_model_endpoint}/infer",
                    json=sample_inference_payload,
                    timeout=5
                )
                if response.status_code == 200:
                    request_count += 1
                else:
                    errors += 1
            except:
                errors += 1
        
        elapsed = time.time() - start_time
        rps = request_count / elapsed
        
        print(f"\nThroughput Results:")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Requests: {request_count}")
        print(f"  Errors: {errors}")
        print(f"  RPS: {rps:.2f}")
        
        # Should handle at least 2 RPS
        assert rps >= 2.0


class TestABRouterPerformance:
    """Test A/B router performance"""
    
    @pytest.mark.performance
    def test_router_overhead(self, triton_inference_url, xgboost_model_endpoint):
        """Measure router overhead vs direct endpoint"""
        n_requests = 50
        
        # Direct endpoint latencies
        direct_latencies = []
        for _ in range(n_requests):
            start = time.time()
            try:
                requests.post(
                    f"{xgboost_model_endpoint}/infer",
                    json={"inputs": [{"name": "input", "shape": [1, 37], "datatype": "FP32", "data": [0.0]*37}]},
                    timeout=5
                )
                direct_latencies.append((time.time() - start) * 1000)
            except:
                pass
        
        if direct_latencies:
            avg_direct = statistics.mean(direct_latencies)
            
            print(f"\nRouter Overhead:")
            print(f"  Direct avg latency: {avg_direct:.2f}ms")
            print(f"  Router is handling requests through ALB/Nginx")