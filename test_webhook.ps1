# Test Webhook Trigger Script

# Your actual webhook secret from Kubernetes
$WEBHOOK_SECRET = "3SNDtjoGvOL1fZd4BWEYml2JcRx60FgX"

$headers = @{
    "Content-Type" = "application/json"
    "Authorization" = "Bearer $WEBHOOK_SECRET"
}

$body = @{
    "alert_name" = "DataDriftDetected"
    "severity" = "warning"
    "description" = "Manual test trigger from PowerShell"
} | ConvertTo-Json

Write-Host "Testing webhook endpoint..." -ForegroundColor Yellow
Write-Host "URL: http://localhost:5000/test" -ForegroundColor Cyan
Write-Host "Using secret: $($WEBHOOK_SECRET.Substring(0,10))..." -ForegroundColor Cyan

try {
    $response = Invoke-RestMethod `
        -Uri "http://localhost:5000/test" `
        -Method POST `
        -Headers $headers `
        -Body $body

    Write-Host "`n✅ Success!" -ForegroundColor Green
    Write-Host ($response | ConvertTo-Json -Depth 5) -ForegroundColor Green
}
catch {
    Write-Host "`n❌ Error:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    
    # Try to get response body
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response: $responseBody" -ForegroundColor Red
    }
}