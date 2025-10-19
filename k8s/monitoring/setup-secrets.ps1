# Setup script for webhook secrets

Write-Host "=== Webhook Secrets Setup ===" -ForegroundColor Cyan

# Get GitHub token
Write-Host "`nEnter your GitHub Personal Access Token:" -ForegroundColor Yellow
Write-Host "(Get from: https://github.com/settings/tokens)" -ForegroundColor Gray
$githubToken = Read-Host -AsSecureString
$githubTokenPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($githubToken))

# Generate webhook secret
Write-Host "`nGenerating webhook secret..." -ForegroundColor Yellow
$webhookSecret = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object {[char]$_})
Write-Host "Generated: $webhookSecret" -ForegroundColor Green

# Create secrets in Kubernetes
Write-Host "`nCreating Kubernetes secrets..." -ForegroundColor Yellow

kubectl delete secret github-secrets -n monitoring 2>$null
kubectl create secret generic github-secrets -n monitoring --from-literal=token="$githubTokenPlain"

kubectl delete secret webhook-secrets -n monitoring 2>$null
kubectl create secret generic webhook-secrets -n monitoring --from-literal=secret="$webhookSecret"

Write-Host "`n✅ Secrets created successfully!" -ForegroundColor Green
Write-Host "`nWebhook Secret: $webhookSecret" -ForegroundColor Cyan
Write-Host "Save this for testing!" -ForegroundColor Yellow

# Clear sensitive variables
$githubTokenPlain = $null