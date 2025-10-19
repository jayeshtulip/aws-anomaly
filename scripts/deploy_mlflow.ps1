# Deploy MLflow on Kubernetes
param(
    [string]$ClusterName = "triton-inference-cluster",
    [string]$AwsRegion = "us-east-1",
    [string]$Namespace = "mlflow"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Deploying MLflow on Kubernetes" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Step 1: Ensure namespace exists
Write-Host "`n[Step 1/6] Creating namespace..." -ForegroundColor Yellow
kubectl create namespace $Namespace --dry-run=client -o yaml | kubectl apply -f -

# Step 2: Deploy PostgreSQL
Write-Host "`n[Step 2/6] Deploying PostgreSQL..." -ForegroundColor Yellow
kubectl apply -f k8s/mlflow/postgres-secret.yaml
kubectl apply -f k8s/mlflow/postgres-statefulset.yaml

Write-Host "Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
kubectl wait --for=condition=ready pod -l app=postgres -n $Namespace --timeout=300s

# Step 3: Verify PostgreSQL
Write-Host "`n[Step 3/6] Verifying PostgreSQL..." -ForegroundColor Yellow
Start-Sleep -Seconds 10
kubectl exec -n $Namespace postgres-0 -- psql -U mlflow -d mlflow_db -c "SELECT 1;"

# Step 4: Deploy MLflow server
Write-Host "`n[Step 4/6] Deploying MLflow server..." -ForegroundColor Yellow
kubectl apply -f k8s/mlflow/mlflow-deployment.yaml

Write-Host "Waiting for MLflow server to be ready..." -ForegroundColor Yellow
kubectl wait --for=condition=ready pod -l app=mlflow-server -n $Namespace --timeout=300s

# Step 5: Deploy Ingress
Write-Host "`n[Step 5/6] Deploying Ingress..." -ForegroundColor Yellow
kubectl apply -f k8s/mlflow/mlflow-ingress.yaml

# Step 6: Get access information
Write-Host "`n[Step 6/6] Getting service information..." -ForegroundColor Yellow

Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "MLflow Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

Write-Host "`nService Status:" -ForegroundColor Cyan
kubectl get pods -n $Namespace

Write-Host "`nLoadBalancer URL:" -ForegroundColor Cyan
$lbHostname = kubectl get svc mlflow-service -n $Namespace -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
Write-Host "http://$lbHostname" -ForegroundColor White

Write-Host "`nIngress URL:" -ForegroundColor Cyan
$ingressHostname = kubectl get ingress mlflow-ingress -n $Namespace -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
if ($ingressHostname) {
    Write-Host "http://$ingressHostname" -ForegroundColor White
}

Write-Host "`nTo access MLflow UI:" -ForegroundColor Cyan
Write-Host "  1. Via LoadBalancer: http://$lbHostname" -ForegroundColor White
Write-Host "  2. Via port-forward: kubectl port-forward -n $Namespace svc/mlflow-service 5000:80" -ForegroundColor White
Write-Host "     Then open: http://localhost:5000" -ForegroundColor White

Write-Host "`n==========================================" -ForegroundColor Green