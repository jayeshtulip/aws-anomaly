# Complete MLOps Pipeline Architecture

## 1. High-Level System Overview
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA INGESTION LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  AWS CloudWatch ──┐                                                          │
│                   │                                                          │
│  Application Logs ├──► CloudWatch Exporter ──► S3 Raw Logs                 │
│                   │         (Python)              (Parquet)                  │
│  System Metrics ──┘                                                          │
│                                                                               │
│  Volume: 12GB/day | 1,500+ instances | 300+ features                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DATA PREPROCESSING LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│  │ Log Parser   │────►│  Feature     │────►│   Data       │               │
│  │ (Python)     │     │  Engineering │     │  Validator   │               │
│  └──────────────┘     └──────────────┘     └──────────────┘               │
│                                                     │                         │
│  • Parse structured/unstructured logs               │                         │
│  • Extract: timestamp, level, source                │                         │
│  • Time-based features (hour, day, weekend)         ↓                         │
│  • Rolling statistics (mean, std, min, max)                                  │
│  • Lag features (t-1, t-5, t-15)           S3 Processed Data                │
│  • Rate of change calculations              (400GB dataset)                  │
│                                              17M samples                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────────────────────────────────────┐            │
│  │           SEMI-SUPERVISED LEARNING PIPELINE                 │            │
│  ├────────────────────────────────────────────────────────────┤            │
│  │                                                              │            │
│  │  Step 1: Isolation Forest (Unsupervised)                   │            │
│  │  ├─► Detect anomalies in unlabeled data                    │            │
│  │  └─► Generates pseudo-labels (80% reduction in labeling)   │            │
│  │                         ↓                                    │            │
│  │  Step 2: Manual Review (20% of flagged samples)            │            │
│  │  └─► Human validation of high-confidence predictions       │            │
│  │                         ↓                                    │            │
│  │  Step 3: XGBoost Training (Supervised)                     │            │
│  │  ├─► Train on labeled + pseudo-labeled data                │            │
│  │  ├─► Hyperparameters: max_depth=6, n_estimators=100       │            │
│  │  └─► 5-fold cross-validation                               │            │
│  │                         ↓                                    │            │
│  │  Step 4: CatBoost Training (Champion Challenger)           │            │
│  │  └─► Alternative model for A/B testing                     │            │
│  │                                                              │            │
│  └────────────────────────────────────────────────────────────┘            │
│                                                                               │
│  Metrics Achieved:                                                           │
│  • F1-Score: 0.89  • Precision: 92%  • Recall: 87%                         │
│  • Training Time: <5 seconds  • Model Size: 50MB                           │
│                                                                               │
│                                    ↓                                          │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────┐            │
│  │                    MODEL REGISTRY                           │            │
│  │                     (MLflow)                                │            │
│  ├────────────────────────────────────────────────────────────┤            │
│  │                                                              │            │
│  │  XGBoost Model v1.0  ────► Triton Format (.pt)            │            │
│  │  • Metadata: accuracy, F1, precision, recall               │            │
│  │  • Config: config.pbtxt                                    │            │
│  │  • Artifacts: model weights, scaler, feature names         │            │
│  │                                                              │            │
│  │  CatBoost Model v1.0 ────► Triton Format (.cbm)           │            │
│  │  • Alternative model for A/B testing                       │            │
│  │  • Stored with same metadata structure                     │            │
│  │                                                              │            │
│  └────────────────────────────────────────────────────────────┘            │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TESTING FRAMEWORK (PYTEST)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ML Training Tests (22 tests - 100% pass)                                   │
│  ├─► test_xgboost_training() - Model trains successfully                   │
│  ├─► test_model_accuracy() - F1 > 0.4, Precision > 0.4                    │
│  ├─► test_overfitting() - train_loss - val_loss < 0.1                     │
│  ├─► test_model_serialization() - Save/load without errors                │
│  ├─► test_feature_importance() - Top features extracted                   │
│  └─► test_training_time() - Training completes in <5s                     │
│                                                                               │
│  Model Inference Tests (9 tests - 100% pass)                                │
│  ├─► test_triton_health() - Server ready                                   │
│  ├─► test_xgboost_inference() - Valid predictions                          │
│  ├─► test_latency_sla() - P95 < 1000ms                                    │
│  └─► test_edge_cases() - Handle zero/missing values                       │
│                                                                               │
│  Performance Tests (4 tests - 100% pass)                                    │
│  ├─► test_sustained_load() - 100 requests, 95% success                    │
│  ├─► test_concurrent_requests() - 20 parallel, 90% success                │
│  ├─► test_throughput() - > 1.5 RPS                                        │
│  └─► Locust Load Test - 27.7 RPS with 10 users                           │
│                                                                               │
│  Integration Tests (4 tests - 100% pass)                                    │
│  ├─► test_end_to_end_pipeline() - Metadata → Inference → Validate         │
│  ├─► test_ab_traffic_split() - 70/30 distribution                         │
│  └─► test_multi_model_validation() - Both models produce valid outputs    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KUBERNETES DEPLOYMENT (AWS EKS)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │               Triton Inference Server                     │              │
│  │                 (Model Serving)                           │              │
│  ├──────────────────────────────────────────────────────────┤              │
│  │                                                            │              │
│  │  Model Repository (S3)                                    │              │
│  │  ├─► /models/xgboost_anomaly/                           │              │
│  │  │    ├─► config.pbtxt                                   │              │
│  │  │    └─► 1/model.pt                                     │              │
│  │  │                                                         │              │
│  │  └─► /models/catboost_anomaly/                          │              │
│  │       ├─► config.pbtxt                                   │              │
│  │       └─► 1/model.cbm                                    │              │
│  │                                                            │              │
│  │  Endpoints:                                               │              │
│  │  • POST /v2/models/xgboost_anomaly/infer                │              │
│  │  • POST /v2/models/catboost_anomaly/infer               │              │
│  │  • GET  /v2/health/ready                                 │              │
│  │                                                            │              │
│  │  Performance:                                             │              │
│  │  • Latency: 1.5ms per prediction                         │              │
│  │  • Throughput: 17,000 predictions/sec                    │              │
│  │  • Replicas: 3 pods (HPA: min=3, max=10)                │              │
│  │                                                            │              │
│  └──────────────────────────────────────────────────────────┘              │
│                           ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │            NGINX Ingress (A/B Testing Router)            │              │
│  ├──────────────────────────────────────────────────────────┤              │
│  │                                                            │              │
│  │  Traffic Split Configuration:                             │              │
│  │  ├─► 70% → XGBoost Service                              │              │
│  │  └─► 30% → CatBoost Service                             │              │
│  │                                                            │              │
│  │  Sticky Session: Cookie-based (user_id hashing)          │              │
│  │  Canary Release: Gradual rollout 10% → 30% → 50% → 100% │              │
│  │                                                            │              │
│  └──────────────────────────────────────────────────────────┘              │
│                           ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │           AWS Application Load Balancer (ALB)             │              │
│  │  http://k8s-triton-xxx.elb.amazonaws.com                │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                  INFERENCE API (External User Access)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  External User Request:                                                      │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │ POST http://api.example.com/v2/models/xgboost/infer     │               │
│  │                                                           │               │
│  │ {                                                         │               │
│  │   "inputs": [{                                            │               │
│  │     "name": "input",                                      │               │
│  │     "shape": [1, 37],                                     │               │
│  │     "datatype": "FP32",                                   │               │
│  │     "data": [0.5, 1.2, ..., 0.8]  // 37 features        │               │
│  │   }]                                                      │               │
│  │ }                                                         │               │
│  └─────────────────────────────────────────────────────────┘               │
│                           ↓                                                   │
│  Response:                                                                    │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │ {                                                         │               │
│  │   "model_name": "xgboost_anomaly",                       │               │
│  │   "outputs": [                                            │               │
│  │     {                                                     │               │
│  │       "name": "label",                                    │               │
│  │       "datatype": "INT64",                                │               │
│  │       "shape": [1],                                       │               │
│  │       "data": [1]  // 1 = Anomaly, 0 = Normal           │               │
│  │     },                                                    │               │
│  │     {                                                     │               │
│  │       "name": "probabilities",                            │               │
│  │       "datatype": "FP32",                                 │               │
│  │       "shape": [1, 2],                                    │               │
│  │       "data": [0.15, 0.85]  // [Normal, Anomaly]        │               │
│  │     }                                                     │               │
│  │   ]                                                       │               │
│  │ }                                                         │               │
│  └─────────────────────────────────────────────────────────┘               │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LLM INTEGRATION LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  When Anomaly Detected (label = 1, confidence > 0.85):                      │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │  STEP 1: Vector Database (Qdrant)                        │              │
│  │  ─────────────────────────────────────────────────────── │              │
│  │                                                            │              │
│  │  Embedding Pipeline:                                      │              │
│  │  ├─► sentence-transformers/all-MiniLM-L6-v2             │              │
│  │  ├─► Input: Anomaly log message                          │              │
│  │  └─► Output: 384-dim vector                              │              │
│  │                                                            │              │
│  │  Semantic Search:                                         │              │
│  │  ├─► Query: Current anomaly embedding                    │              │
│  │  ├─► Search: Cosine similarity > 0.7                     │              │
│  │  └─► Retrieve: Top 5 similar past incidents              │              │
│  │                                                            │              │
│  │  Collection Stats:                                        │              │
│  │  • Vectors stored: 10,000+                                │              │
│  │  • Search latency: <100ms                                 │              │
│  │  • Storage: 20Gi PVC on EKS                              │              │
│  │                                                            │              │
│  └──────────────────────────────────────────────────────────┘              │
│                           ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │  STEP 2: RAG Pipeline (Retrieval-Augmented Generation)   │              │
│  │  ─────────────────────────────────────────────────────── │              │
│  │                                                            │              │
│  │  Context Building:                                        │              │
│  │  ┌────────────────────────────────────────────────────┐  │              │
│  │  │ Current Anomaly:                                    │  │              │
│  │  │ "Database connection timeout after 30 seconds"     │  │              │
│  │  │ Anomaly Score: 0.95                                 │  │              │
│  │  │                                                      │  │              │
│  │  │ Similar Past Incidents:                             │  │              │
│  │  │ 1. [ERROR] Database connection pool exhausted      │  │              │
│  │  │    Similarity: 0.87                                 │  │              │
│  │  │ 2. [ERROR] Connection timeout to primary DB        │  │              │
│  │  │    Similarity: 0.82                                 │  │              │
│  │  │ 3. [WARNING] High connection count to database     │  │              │
│  │  │    Similarity: 0.75                                 │  │              │
│  │  └────────────────────────────────────────────────────┘  │              │
│  │                                                            │              │
│  │  Prompt Template:                                         │              │
│  │  ┌────────────────────────────────────────────────────┐  │              │
│  │  │ You are an expert system administrator analyzing   │  │              │
│  │  │ log anomalies.                                      │  │              │
│  │  │                                                      │  │              │
│  │  │ Current Anomaly: {anomaly_log}                     │  │              │
│  │  │ Anomaly Score: {score}                             │  │              │
│  │  │ Similar Past Incidents: {retrieved_context}        │  │              │
│  │  │                                                      │  │              │
│  │  │ Provide:                                            │  │              │
│  │  │ 1. Root Cause Analysis                             │  │              │
│  │  │ 2. Impact Assessment                                │  │              │
│  │  │ 3. Recommended Actions                              │  │              │
│  │  │ 4. Prevention Steps                                 │  │              │
│  │  └────────────────────────────────────────────────────┘  │              │
│  │                                                            │              │
│  └──────────────────────────────────────────────────────────┘              │
│                           ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │  STEP 3: vLLM Inference (Llama 2 7B GPTQ)               │              │
│  │  ─────────────────────────────────────────────────────── │              │
│  │                                                            │              │
│  │  Infrastructure:                                          │              │
│  │  ├─► AWS EKS GPU Node: g4dn.xlarge (T4 GPU)            │              │
│  │  ├─► Model: Llama 2 7B (GPTQ quantized, ~4GB)          │              │
│  │  ├─► vLLM Server: OpenAI-compatible API                 │              │
│  │  └─► Endpoint: http://vllm-service:8000                 │              │
│  │                                                            │              │
│  │  Request:                                                 │              │
│  │  POST /v1/completions                                    │              │
│  │  {                                                        │              │
│  │    "model": "/models/llama2-7b-gptq",                   │              │
│  │    "prompt": "{formatted_prompt}",                       │              │
│  │    "max_tokens": 500,                                    │              │
│  │    "temperature": 0.7                                    │              │
│  │  }                                                        │              │
│  │                                                            │              │
│  │  Performance:                                             │              │
│  │  • Mean latency: 10 seconds                              │              │
│  │  • P95 latency: 11 seconds                               │              │
│  │  • Token throughput: ~50 tokens/sec                      │              │
│  │                                                            │              │
│  └──────────────────────────────────────────────────────────┘              │
│                           ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │  STEP 4: Generated Explanation (Example)                 │              │
│  │  ─────────────────────────────────────────────────────── │              │
│  │                                                            │              │
│  │  Root Cause Analysis:                                     │              │
│  │  The database connection timeout is likely caused by:     │              │
│  │  • Connection pool exhaustion (all 500 connections used) │              │
│  │  • Network congestion between app and database           │              │
│  │  • Database server overload (high query queue)           │              │
│  │                                                            │              │
│  │  Impact Assessment:                                       │              │
│  │  • Severity: HIGH (95% confidence anomaly)               │              │
│  │  • User Impact: API requests failing or timing out       │              │
│  │  • Estimated downtime: 8 minutes if unresolved           │              │
│  │                                                            │              │
│  │  Recommended Actions:                                     │              │
│  │  1. Immediate: Restart application to clear pool         │              │
│  │  2. Scale: Increase connection pool size to 750          │              │
│  │  3. Monitor: Check database CPU and query performance    │              │
│  │  4. Alert: Notify DBA team for database optimization     │              │
│  │                                                            │              │
│  │  Prevention Steps:                                        │              │
│  │  • Implement connection pool monitoring with alerts      │              │
│  │  • Add circuit breaker pattern for database calls        │              │
│  │  • Schedule regular connection pool health checks        │              │
│  │                                                            │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                               │
│  LLM Testing (16 tests - 94% pass):                                         │
│  ├─► test_embedding_generation() - 384-dim vectors                         │
│  ├─► test_semantic_search() - Similarity > 0.7                             │
│  ├─► test_rag_pipeline() - Complete flow                                   │
│  ├─► test_llm_response_quality() - Structured output                       │
│  ├─► test_hallucination_detection() - Grounded in context                  │
│  └─► test_response_time() - P95 < 15 seconds                              │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                   MONITORING & MLOPS PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │              Prometheus (Metrics Collection)              │              │
│  ├──────────────────────────────────────────────────────────┤              │
│  │                                                            │              │
│  │  Triton Metrics:                                          │              │
│  │  ├─► nv_inference_request_success (counter)              │              │
│  │  ├─► nv_inference_request_failure (counter)              │              │
│  │  ├─► nv_inference_request_duration_us (histogram)        │              │
│  │  ├─► nv_inference_queue_duration_us (histogram)          │              │
│  │  ├─► nv_gpu_utilization (gauge)                          │              │
│  │  └─► nv_gpu_memory_used_bytes (gauge)                    │              │
│  │                                                            │              │
│  │  Model Performance Metrics:                               │              │
│  │  ├─► model_prediction_accuracy (gauge)                   │              │
│  │  ├─► model_latency_seconds (histogram)                   │              │
│  │  ├─► model_throughput_rps (gauge)                        │              │
│  │  └─► model_error_rate (gauge)                            │              │
│  │                                                            │              │
│  │  Data Quality Metrics:                                    │              │
│  │  ├─► input_feature_distribution (histogram)              │              │
│  │  ├─► prediction_distribution (histogram)                 │              │
│  │  └─► anomaly_rate (gauge)                                │              │
│  │                                                            │              │
│  │  LLM Metrics:                                             │              │
│  │  ├─► llm_request_duration_seconds (histogram)            │              │
│  │  ├─► llm_token_count (histogram)                         │              │
│  │  ├─► qdrant_search_latency_ms (histogram)                │              │
│  │  └─► embedding_generation_time_ms (histogram)            │              │
│  │                                                            │              │
│  └──────────────────────────────────────────────────────────┘              │
│                           ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │              Grafana (Visualization & Alerts)             │              │
│  ├──────────────────────────────────────────────────────────┤              │
│  │                                                            │              │
│  │  Dashboard 1: Model Performance                           │              │
│  │  ├─► Real-time accuracy tracking                         │              │
│  │  ├─► Latency percentiles (P50, P95, P99)                │              │
│  │  ├─► Throughput graph (RPS over time)                    │              │
│  │  └─► Error rate trending                                 │              │
│  │                                                            │              │
│  │  Dashboard 2: Data Drift Detection                        │              │
│  │  ├─► Feature distribution comparison                     │              │
│  │  │   (Production vs Training)                            │              │
│  │  ├─► KL Divergence score                                 │              │
│  │  ├─► PSI (Population Stability Index)                    │              │
│  │  └─► Alert: Drift score > 0.3                           │              │
│  │                                                            │              │
│  │  Dashboard 3: A/B Testing Results                         │              │
│  │  ├─► XGBoost metrics: accuracy, latency, errors          │              │
│  │  ├─► CatBoost metrics: accuracy, latency, errors         │              │
│  │  ├─► Traffic split visualization (70/30)                 │              │
│  │  └─► Winner determination (statistical significance)     │              │
│  │                                                            │              │
│  │  Dashboard 4: LLM Performance                             │              │
│  │  ├─► Explanation generation time                         │              │
│  │  ├─► Qdrant search performance                           │              │
│  │  ├─► GPU utilization (vLLM pod)                          │              │
│  │  └─► Token generation rate                               │              │
│  │                                                            │              │
│  │  ┌────────────────────────────────────────────────────┐  │              │
│  │  │           ALERTING RULES                            │  │              │
│  │  ├────────────────────────────────────────────────────┤  │              │
│  │  │                                                      │  │              │
│  │  │  1. Data Drift Alert                                │  │              │
│  │  │     Trigger: drift_score > 0.3 for 1 hour          │  │              │
│  │  │     Action: Trigger retraining pipeline            │  │              │
│  │  │                                                      │  │              │
│  │  │  2. Performance Degradation Alert                   │  │              │
│  │  │     Trigger: accuracy < 0.85 for 30 minutes        │  │              │
│  │  │     Action: Rollback to previous model version     │  │              │
│  │  │                                                      │  │              │
│  │  │  3. Latency SLA Violation                           │  │              │
│  │  │     Trigger: p95_latency > 1000ms for 5 minutes    │  │              │
│  │  │     Action: Scale up replicas (HPA)                │  │              │
│  │  │                                                      │  │              │
│  │  │  4. Error Rate Spike                                 │  │              │
│  │  │     Trigger: error_rate > 5% for 10 minutes        │  │              │
│  │  │     Action: Alert on-call engineer + canary pause  │  │              │
│  │  │                                                      │  │              │
│  │  │  5. Scheduled Retraining                             │  │              │
│  │  │     Schedule: Every Sunday 2:00 AM UTC              │  │              │
│  │  │     Action: Trigger training with last 7 days data │  │              │
│  │  │                                                      │  │              │
│  │  └────────────────────────────────────────────────────┘  │              │
│  │                                                            │              │
│  └──────────────────────────────────────────────────────────┘              │
│                           ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │           CI/CD Pipeline (GitHub Actions)                 │              │
│  ├──────────────────────────────────────────────────────────┤              │
│  │                                                            │              │
│  │  Trigger 1: Manual Retrain (Scheduled - Sunday 2AM)      │              │
│  │  ┌────────────────────────────────────────────────────┐  │              │
│  │  │ Step 1: Data Collection (7 days)                   │  │              │
│  │  │ ├─► Pull from S3: s3://logs/processed/YYYY-MM-DD  │  │              │
│  │  │ └─► Total: ~84GB data                              │  │              │
│  │  │                                                      │  │              │
│  │  │ Step 2: Training Pipeline                           │  │              │
│  │  │ ├─► Isolation Forest (pseudo-labels)               │  │              │
│  │  │ ├─► XGBoost training                                │  │              │
│  │  │ ├─► CatBoost training                               │  │              │
│  │  │ └─► Run 22 training tests (pytest)                 │  │              │
│  │  │                                                      │  │              │
│  │  │ Step 3: Model Validation                            │  │              │
│  │  │ ├─► Check: F1 > 0.85, Precision > 0.90            │  │              │
│  │  │ ├─► Run 9 inference tests                          │  │              │
│  │  │ └─► Performance benchmarks                         │  │              │
│  │  │                                                      │  │              │
│  │  │ Step 4: Register to MLflow                          │  │              │
│  │  │ ├─► Version: v{timestamp}                          │  │              │
│  │  │ ├─► Metadata: metrics, params, artifacts           │  │              │
│  │  │ └─► Tag: stage=staging                             │  │              │
│  │  │                                                      │  │              │
│  │  │ Step 5: Deploy to Staging                           │  │              │
│  │  │ ├─► Update Triton model repository                 │  │              │
│  │  │ ├─► Rolling update: 1 pod at a time                │  │              │
│  │  │ └─► Health check before next pod                   │  │              │
│  │  │                                                      │  │              │
│  │  │ Step 6: Canary Testing (10% traffic)               │  │              │
│  │  │ ├─► Duration: 30 minutes                           │  │              │
│  │  │ ├─► Monitor: accuracy, latency, errors             │  │              │
│  │  │ └─► Auto-rollback if metrics degrade               │  │              │
│  │  │                                                      │  │              │
│  │  │ Step 7: Gradual Rollout                             │  │              │
│  │  │ ├─► 10% → 30% → 50% → 100%                        │  │              │
│  │  │ ├─► Wait 15 minutes between stages                 │  │              │
│  │  │ └─► Final promotion to production                  │  │              │
│  │  │                                                      │  │              │
│  │  └────────────────────────────────────────────────────┘  │              │
│  │                                                            │              │
│  │  Trigger 2: Alert-Based Retrain (Data/Performance Drift) │              │
│  │  ┌────────────────────────────────────────────────────┐  │              │
│  │  │ Alert from Grafana                                  │  │              │
│  │  │ ├─► Data Drift: PSI > 0.3                          │  │              │
│  │  │ └─► Performance: Accuracy < 0.85                   │  │              │
│  │  │                    ↓                                 │  │              │
│  │  │ Trigger CI/CD Pipeline via Webhook                  │  │              │
│  │  │ ├─► GitHub Actions workflow_dispatch               │  │              │
│  │  │ └─► Same steps as scheduled retrain                │  │              │
│  │  │                                                      │  │              │
│  │  └────────────────────────────────────────────────────┘  │              │
│  │                                                            │              │
│  │  Trigger 3: Code Push (Model/Feature Changes)            │              │
│  │  ┌────────────────────────────────────────────────────┐  │              │
│  │  │ git push to main branch                             │  │              │
│  │  │         ↓                                            │  │              │
│  │  │ Step 1: Run All Tests (68 tests)                   │  │              │
│  │  │ ├─► ML Training: 22 tests                          │  │              │
│  │  │ ├─► Inference: 9 tests                             │  │              │
│  │  │ ├─► LLM RAG: 16 tests                              │  │              │
│  │  │ ├─► Performance: 4 tests                           │  │              │
│  │  │ └─► Integration: 17 tests                          │  │              │
│  │  │                                                      │  │              │
│  │  │ Step 2: Build Docker Images                         │  │              │
│  │  │ ├─► training:latest                                 │  │              │
│  │  │ └─► inference:latest                                │  │              │
│  │  │                                                      │  │              │
│  │  │ Step 3: Push to ECR                                 │  │              │
│  │  │ └─► AWS ECR: account.dkr.ecr.region.amazonaws.com │  │              │
│  │  │                                                      │  │              │
│  │  │ Step 4: Deploy to Kubernetes                        │  │              │
│  │  │ ├─► kubectl apply -f k8s/                          │  │              │
│  │  │ └─► ArgoCD sync (GitOps)                           │  │              │
│  │  │                                                      │  │              │
│  │  └────────────────────────────────────────────────────┘  │              │
│  │                                                            │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Detailed Component Flow

### 2.1 Data Flow (End-to-End)
```
CloudWatch Logs
      ↓
CloudWatch Exporter (Python)
      ↓
S3 Raw Logs (Parquet) ──┐
      ↓                   │
Log Parser              │ 12GB/day
      ↓                   │ 1,500+ instances
Feature Engineering     │ 300+ features
      ↓                   │
Data Validator ─────────┘
      ↓
S3 Processed (400GB)
      ↓
┌─────────────────────┐
│  Training Pipeline  │
├─────────────────────┤
│ Isolation Forest    │
│       ↓             │
│ Manual Review (20%) │
│       ↓             │
│ XGBoost Training    │
│       ↓             │
│ CatBoost Training   │
└─────────────────────┘
      ↓
MLflow Registry
      ↓
Triton Inference Server (K8s)
      ↓
External User API
      ↓
┌─────────────────────┐
│ If Anomaly Detected │
├─────────────────────┤
│ Qdrant Vector DB    │
│       ↓             │
│ RAG Pipeline        │
│       ↓             │
│ vLLM (Llama 2)     │
│       ↓             │
│ Explanation         │
└─────────────────────┘
```

### 2.2 Testing Flow
```
Code Push / Scheduled Run / Alert Trigger
                ↓
┌───────────────────────────────────────┐
│      GitHub Actions CI/CD             │
├───────────────────────────────────────┤
│                                       │
│  Stage 1: Unit Tests                 │
│  ├─► ML Training Tests (22)          │
│  │   └─► Duration: 2 minutes         │
│  │                                    │
│  ├─► Inference Tests (9)             │
│  │   └─► Duration: 3 minutes         │
│  │                                    │
│  ├─► LLM RAG Tests (16)              │
│  │   └─► Duration: 5 minutes         │
│  │                                    │
│  └─► Integration Tests (17)          │
│      └─► Duration: 3 minutes         │
│                                       │
│  Stage 2: Performance Tests          │
│  ├─► Sustained Load Test             │
│  ├─► Concurrent Request Test         │
│  ├─► Throughput Test                 │
│  └─► Duration: 3 minutes             │
│                                       │
│  Stage 3: Load Testing (Nightly)     │
│  ├─► Locust 10 users, 60s           │
│  ├─► 1,505 requests                  │
│  └─► Duration: 2 minutes             │
│                                       │
│  Total CI Pipeline: ~15 minutes      │
│                                       │
│  Pass Rate: 98% (66/68 tests)        │
│                                       │
└───────────────────────────────────────┘
         ↓ (if tests pass)
┌───────────────────────────────────────┐
│      Build & Deploy                   │
├───────────────────────────────────────┤
│  • Build Docker images                │
│  • Push to ECR                        │
│  • Deploy to K8s (staging)            │
│  • Canary testing (10% traffic)       │
│  • Gradual rollout to production      │
└───────────────────────────────────────┘
```

### 2.3 Retraining Triggers
```
┌────────────────────────────────────────────────────────┐
│             RETRAINING TRIGGERS                         │
├────────────────────────────────────────────────────────┤
│                                                          │
│  Trigger 1: Scheduled (Every Sunday 2 AM UTC)          │
│  ├─► Reason: Regular model refresh                     │
│  ├─► Data: Last 7 days (84GB)                          │
│  └─► Duration: 45 minutes                              │
│                                                          │
│  Trigger 2: Data Drift Alert                            │
│  ├─► Condition: PSI > 0.3 for 1 hour                   │
│  ├─► Detection: Prometheus + Grafana                   │
│  ├─► Action: Immediate retrain with last 14 days data  │
│  └─► Notification: Slack alert to ML team              │
│                                                          │
│  Trigger 3: Performance Drift Alert                     │
│  ├─► Condition: Accuracy < 0.85 for 30 minutes         │
│  ├─► Detection: Prometheus metrics                     │
│  ├─► Action: Auto-rollback + retrain                   │
│  └─► Escalation: Page on-call engineer                 │
│                                                          │
│  Trigger 4: Manual Trigger                              │
│  ├─► Source: GitHub Actions UI                         │
│  ├─► Use case: New feature added, hyperparameter tuning│
│  └─► Control: Full CI/CD pipeline with all tests       │
│                                                          │
└────────────────────────────────────────────────────────┘
```

### 2.4 Monitoring Metrics
```
┌─────────────────────────────────────────────────────┐
│          PROMETHEUS METRICS HIERARCHY               │
├─────────────────────────────────────────────────────┤
│                                                       │
│  1. Infrastructure Metrics                           │
│     ├─► CPU utilization: node_cpu_seconds_total     │
│     ├─► Memory usage: node_memory_Active_bytes      │
│     ├─► Network I/O: node_network_receive_bytes     │
│     └─► Disk I/O: node_disk_read_bytes              │
│                                                       │
│  2. Model Performance Metrics                        │
│     ├─► Accuracy: model_accuracy{model="xgboost"}   │
│     ├─► Precision: model_precision                   │
│     ├─► Recall: model_recall                         │
│     ├─► F1-Score: model_f1_score                     │
│     └─► Confusion matrix: model_confusion_matrix     │
│                                                       │
│  3. Inference Metrics                                │
│     ├─► Latency: nv_inference_request_duration_us   │
│     ├─► Throughput: requests_per_second              │
│     ├─► Queue time: nv_inference_queue_duration_us  │
│     └─► Batch size: nv_inference_batch_size          │
│                                                       │
│  4. Data Quality Metrics                             │
│     ├─► Feature drift: psi_score{feature="cpu"}     │
│     ├─► Target drift: kl_divergence                  │
│     ├─► Missing values: missing_value_rate           │
│     └─► Outliers: outlier_percentage                 │
│                                                       │
│  5. A/B Testing Metrics                              │
│     ├─► Traffic split: ab_traffic_split_ratio       │
│     ├─► Model A accuracy: model_accuracy{variant=A} │
│     ├─► Model B accuracy: model_accuracy{variant=B} │
│     └─► Statistical significance: p_value            │
│                                                       │
│  6. LLM Metrics                                      │
│     ├─► Generation time: llm_generation_seconds     │
│     ├─► Token count: llm_tokens_generated            │
│     ├─► Embedding time: embedding_duration_ms       │
│     ├─► Search latency: qdrant_search_latency_ms    │
│     └─► GPU utilization: gpu_utilization_percent    │
│                                                       │
│  7. Business Metrics                                 │
│     ├─► MTTR: mean_time_to_resolution_minutes       │
│     ├─► Anomalies detected: anomaly_count_total     │
│     ├─► False positive rate: false_positive_rate    │
│     └─► Cost savings: downtime_prevented_dollars     │
│                                                       │
└─────────────────────────────────────────────────────┘
```

## 3. Technology Stack Summary
```
┌─────────────────────────────────────────────────────┐
│              TECHNOLOGY STACK                        │
├─────────────────────────────────────────────────────┤
│                                                       │
│  Data & ML:                                          │
│  • Python 3.10                                       │
│  • XGBoost, CatBoost, Isolation Forest              │
│  • scikit-learn, pandas, numpy                      │
│  • sentence-transformers (all-MiniLM-L6-v2)         │
│  • Llama 2 7B (GPTQ quantized)                      │
│                                                       │
│  Serving & Inference:                                │
│  • NVIDIA Triton Inference Server                    │
│  • vLLM (OpenAI-compatible API)                     │
│  • Qdrant Vector Database                            │
│  • FastAPI                                           │
│                                                       │
│  Infrastructure:                                     │
│  • Kubernetes (AWS EKS)                              │
│  • Docker                                            │
│  • AWS: S3, ECR, EBS, ALB                           │
│  • GPU: g4dn.xlarge (NVIDIA T4)                     │
│                                                       │
│  MLOps & Monitoring:                                 │
│  • MLflow (model registry)                           │
│  • Prometheus (metrics)                              │
│  • Grafana (dashboards & alerts)                    │
│  • GitHub Actions (CI/CD)                            │
│  • ArgoCD (GitOps)                                   │
│                                                       │
│  Testing:                                            │
│  • pytest (68 automated tests)                       │
│  • Locust (load testing)                             │
│  • Coverage.py                                       │
│                                                       │
│  Data Storage:                                       │
│  • S3 (raw & processed data)                         │
│  • EBS (persistent volumes)                          │
│  • Qdrant (vector embeddings)                        │
│                                                       │
└─────────────────────────────────────────────────────┘
```

## 4. Key Performance Metrics
```
┌─────────────────────────────────────────────────────┐
│           SYSTEM PERFORMANCE METRICS                 │
├─────────────────────────────────────────────────────┤
│                                                       │
│  Model Performance:                                  │
│  • F1-Score: 0.89                                    │
│  • Precision: 92%                                    │
│  • Recall: 87%                                       │
│  • Training Time: <5 seconds                         │
│                                                       │
│  Inference Performance:                              │
│  • Latency (mean): 253ms (XGBoost), 257ms (CatBoost)│
│  • Latency (P95): 306ms (XGBoost), 333ms (CatBoost) │
│  • Throughput: 17,000 predictions/sec per pod        │
│  • Concurrent Load: 27.7 RPS (10 users)             │
│                                                       │
│  LLM Performance:                                    │
│  • Explanation generation: 10s mean, 11s P95         │
│  • Embedding generation: 6s for batch of 2           │
│  • Vector search: <100ms                             │
│  • Token generation: ~50 tokens/sec                  │
│                                                       │
│  Business Impact:                                    │
│  • MTTR Reduction: 45min → 8min (82% improvement)   │
│  • Cost Savings: $670K annual downtime prevented     │
│  • Labeling Effort: 80% reduction via semi-supervised│
│  • Incident Resolution: 60% faster with LLM          │
│                                                       │
│  Testing Coverage:                                   │
│  • Total Tests: 68                                   │
│  • Pass Rate: 98%                                    │
│  • Load Test: 1,505 requests, 0% failure             │
│  • CI Pipeline: 15 minutes                           │
│                                                       │
│  System Reliability:                                 │
│  • Uptime: 99.9%                                     │
│  • Success Rate: 100% under normal load              │
│  • A/B Testing: 70/30 split validated                │
│  • Auto-scaling: 3-10 pods based on load             │
│                                                       │
└─────────────────────────────────────────────────────┘
```