# Testing Documentation

## Overview
Comprehensive test suite for production ML inference system with 39 automated tests covering training, inference, performance, and integration scenarios.

## Test Architecture
```
tests/
├── unit/
│   ├── test_model_training.py      # 22 tests - Model training & validation
│   └── test_model_inference.py     # 9 tests - Live endpoint testing
├── performance/
│   ├── test_performance.py         # 4 tests - Load, throughput, concurrency
│   └── locustfile.py              # Load testing scenarios
└── integration/
    └── test_full_pipeline.py       # 4 tests - E2E workflows
```

## Quick Start

### Run All Tests
```bash
pytest tests/ -v
```

### Run by Category
```bash
# Training tests only
pytest tests/unit/test_model_training.py -v

# Inference tests only
pytest tests/unit/test_model_inference.py -v

# Performance tests
pytest tests/performance/test_performance.py -v -m performance

# Integration tests
pytest tests/integration/test_full_pipeline.py -v
```

### Load Testing
```bash
# 60-second load test with 10 users
locust -f tests/performance/locustfile.py \
  --host=http://k8s-triton-tritonab-aafed81da0-1339382983.us-east-1.elb.amazonaws.com \
  --headless --users 10 --spawn-rate 2 --run-time 60s

# Interactive mode (with web UI on port 8089)
locust -f tests/performance/locustfile.py \
  --host=http://k8s-triton-tritonab-aafed81da0-1339382983.us-east-1.elb.amazonaws.com
```

## Test Categories

### 1. Unit Tests - Model Training (22 tests)
**File:** `tests/unit/test_model_training.py`

- ✅ XGBoost training and validation
- ✅ CatBoost training and validation  
- ✅ Isolation Forest training
- ✅ Model serialization/deserialization
- ✅ Hyperparameter validation
- ✅ Overfitting detection
- ✅ Feature importance extraction
- ✅ Model metadata persistence
- ✅ Configuration validation
- ✅ Performance benchmarks

**Key Assertions:**
- Models train successfully without errors
- Accuracy exceeds threshold (>40% on random data)
- Model files save and load correctly
- Training completes in reasonable time (<5s)
- Memory usage stays under limits

### 2. Unit Tests - Model Inference (9 tests)
**File:** `tests/unit/test_model_inference.py`

**Triton Connection (2 tests):**
- Server liveness check
- Server readiness check

**XGBoost Inference (5 tests):**
- Model availability check
- Valid input inference
- Output format validation
- Prediction consistency
- Latency SLA (<1000ms)

**CatBoost Inference (2 tests):**
- Model availability check
- Valid input inference

**Edge Cases (1 test):**
- Inference with zero values

**Key Assertions:**
- Models are loaded and accessible
- Responses follow Triton v2 protocol
- Latency meets SLA requirements
- Predictions are deterministic
- Edge cases handled gracefully

### 3. Performance Tests (4 tests)
**File:** `tests/performance/test_performance.py`

**Load Tests:**
- ✅ Sustained load (100 sequential requests)
  - 95% success rate required
  - P95 latency < 1000ms
  - Mean latency < 500ms

- ✅ Concurrent requests (20 parallel)
  - 90% success rate under load
  - Mean latency < 2000ms
  
- ✅ Throughput measurement (10-second test)
  - Minimum 2 requests/second

- ✅ Router overhead analysis
  - Compare direct vs routed latency

**Metrics Collected:**
- Mean, median, P95, P99 latencies
- Success/failure rates
- Requests per second (RPS)
- Error rates under load

### 4. Integration Tests (4 tests)
**File:** `tests/integration/test_full_pipeline.py`

**End-to-End Workflow:**
- ✅ Model metadata → Inference → Response validation
- ✅ Complete prediction pipeline
- ✅ Output validation (probabilities sum to 1)

**A/B Testing:**
- ✅ Traffic split verification (70/30)
- ✅ Both models accessible
- ✅ Both models produce valid predictions

**Model Versioning:**
- ✅ Version metadata available
- ✅ Specific version accessible

### 5. Load Testing with Locust
**File:** `tests/performance/locustfile.py`

**User Scenarios:**
- `TritonInferenceUser`: Simulates normal traffic
  - 70% XGBoost requests
  - 30% CatBoost requests
  - Periodic health checks
  - Model metadata queries

- `StressTestUser`: Stress testing
  - Rapid-fire requests (10-50ms intervals)
  - Maximum throughput testing

**Metrics:**
- Real-time RPS (requests per second)
- Response time distribution
- Failure rate
- Concurrent user handling

## Test Results

### Current Performance Benchmarks

| Metric | XGBoost | CatBoost |
|--------|---------|----------|
| Avg Latency | 256ms | 327ms |
| P95 Latency | 264ms | ~350ms |
| P99 Latency | <400ms | <500ms |
| Success Rate | 100% | 100% |
| Throughput | 3-4 RPS | 2-3 RPS |

### Load Test Results (10 users, 60s)
- **Total Requests:** ~180
- **Success Rate:** >95%
- **Mean Latency:** ~300ms
- **Max Latency:** <2000ms
- **No failures under normal load**

## CI/CD Integration

### GitHub Actions Example
```yaml
name: ML Model Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov locust
    - name: Run unit tests
      run: pytest tests/unit/ -v --cov=.
    - name: Run integration tests
      run: pytest tests/integration/ -v
```

## Troubleshooting

### Tests Failing?

**Check Endpoint Availability:**
```bash
curl http://k8s-triton-tritonab-aafed81da0-1339382983.us-east-1.elb.amazonaws.com/v2/health/ready
```

**Check Models Loaded:**
```bash
curl http://k8s-triton-tritonab-aafed81da0-1339382983.us-east-1.elb.amazonaws.com/v2/models/xgboost_anomaly
curl http://k8s-triton-tritonab-aafed81da0-1339382983.us-east-1.elb.amazonaws.com/v2/models/catboost_anomaly
```

**Check Kubernetes Pods:**
```bash
kubectl get pods -n triton
kubectl logs -n triton -l app=triton
```

### Performance Issues?

**Scale Up Replicas:**
```bash
kubectl scale deployment triton-server -n triton --replicas=5
```

**Check Resource Usage:**
```bash
kubectl top pods -n triton
```

## Future Enhancements

- [ ] Add data drift detection tests
- [ ] Add model accuracy regression tests
- [ ] Add security/authentication tests
- [ ] Add chaos engineering tests
- [ ] Add multi-region failover tests
- [ ] Add model explainability tests (SHAP values)

## Contributing

When adding new tests:
1. Follow existing test structure
2. Add appropriate markers (`@pytest.mark.performance`, etc.)
3. Include docstrings
4. Update this README
5. Ensure tests are idempotent

## License
MIT License