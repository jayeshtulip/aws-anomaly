#!/bin/bash

# Deploy MLflow on Kubernetes
set -e

CLUSTER_NAME="triton-inference-cluster"
AWS_REGION="us-east-1"
NAMESPACE="mlflow"

echo "=========================================="
echo "Deploying MLflow on Kubernetes"
echo "=========================================="

# Step 1: Ensure namespace exists
echo "[Step 1/6] Creating namespace..."
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Step 2: Deploy PostgreSQL
echo "[Step 2/6] Deploying PostgreSQL..."
kubectl apply -f k8s/mlflow/postgres-secret.yaml
kubectl apply -f k8s/mlflow/postgres-statefulset.yaml

echo "Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n ${NAMESPACE} --timeout=300s

# Step 3: Verify PostgreSQL connection
echo "[Step 3/6] Verifying PostgreSQL..."
kubectl exec -n ${NAMESPACE} postgres-0 -- psql -U mlflow -d mlflow_db -c "SELECT 1;" || echo "PostgreSQL check completed"

# Step 4: Deploy MLflow server
echo "[Step 4/6] Deploying MLflow server..."
kubectl apply -f k8s/mlflow/mlflow-deployment.yaml

echo "Waiting for MLflow server to be ready..."
kubectl wait --for=condition=ready pod -l app=mlflow-server -n ${NAMESPACE} --timeout=300s

# Step 5: Deploy Ingress (optional - comment out if using LoadBalancer only)
echo "[Step 5/6] Deploying Ingress..."
kubectl apply -f k8s/mlflow/mlflow-ingress.yaml

# Step 6: Get access information
echo "[Step 6/6] Getting service information..."
echo ""
echo "=========================================="
echo "MLflow Deployment Complete!"
echo "=========================================="
echo ""
echo "Service Status:"
kubectl get pods -n ${NAMESPACE}
echo ""
echo "LoadBalancer URL:"
kubectl get svc mlflow-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
echo ""
echo ""
echo "Ingress URL (if deployed):"
kubectl get ingress mlflow-ingress -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
echo ""
echo ""
echo "To access MLflow UI:"
echo "  1. Via LoadBalancer: http://$(kubectl get svc mlflow-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')"
echo "  2. Via port-forward: kubectl port-forward -n ${NAMESPACE} svc/mlflow-service 5000:80"
echo ""
echo "=========================================="