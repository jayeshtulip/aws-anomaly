# AWS Event Anomaly Detection - MLOps Pipeline

End-to-end MLOps pipeline for AWS CloudWatch log anomaly detection with ML models and LLM integration.

## Project Structure
```
aws-anomaly-detection/
├── data_ingestion/          # CloudWatch log ingestion
├── data_preprocessing/      # Data cleaning and feature engineering
├── models/                  # ML model training code
│   ├── isolation_forest/    # Unsupervised anomaly detection
│   ├── xgboost/            # Supervised classification
│   └── catboost/           # Alternative model for A/B testing
├── llm/                    # LLM integration and RAG pipeline
├── ray_serve/              # Ray Serve deployments
├── ab_testing/             # A/B testing framework
├── k8s/                    # Kubernetes manifests
├── triton_models/          # Triton model repository
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── notebooks/              # Jupyter notebooks for exploration
└── config/                 # Configuration files
```

## Prerequisites

- Python 3.9+
- AWS CLI configured
- kubectl configured with EKS cluster
- Docker Desktop (for local testing)
- VS Code with Python extension

## AWS Resources

- **EKS Cluster**: triton-inference-cluster
- **S3 Bucket**: triton-models-71544
- **IAM User**: Jayesharnav
- **AWS Account**: 755289151343

## Quick Start

1. **Clone and setup environment**:
```bash
   git clone <repository-url>
   cd aws-anomaly-detection
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
```

2. **Configure AWS credentials**:
```bash
   aws configure
```

3. **Initialize DVC**:
```bash
   dvc init
   dvc remote add -d myremote s3://triton-models-71544/dvc-storage
```

4. **Run tests**:
```bash
   pytest tests/
```

## Development Workflow

1. Data preprocessing
2. Model training with MLflow tracking
3. Model evaluation and A/B testing
4. Model deployment to Triton/Ray Serve
5. LLM integration for explanations
6. Monitoring and alerting

## Documentation

- [Setup Guide](docs/setup.md)
- [Model Training](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [API Documentation](docs/api.md)

## License

MIT License