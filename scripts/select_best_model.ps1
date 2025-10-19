# Select Best Model for Deployment
param(
    [string]$Criterion = "f1_score"  # Options: f1_score, accuracy, auc_roc, precision, recall
)

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "MODEL SELECTION FOR DEPLOYMENT" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan

# Load model comparison
$comparison = Get-Content metrics/model_comparison.json | ConvertFrom-Json

Write-Host "`n[1] All Models Performance:" -ForegroundColor Yellow
Write-Host ""
$comparison | Format-Table model, accuracy, precision, recall, f1_score, auc_roc -AutoSize

# Select best model based on criterion
$bestModel = $comparison | Sort-Object $Criterion -Descending | Select-Object -First 1

Write-Host "`n[2] Best Model (by $Criterion):" -ForegroundColor Yellow
Write-Host ""
Write-Host "Model:     $($bestModel.model)" -ForegroundColor Green
Write-Host "Accuracy:  $($bestModel.accuracy)" -ForegroundColor White
Write-Host "Precision: $($bestModel.precision)" -ForegroundColor White
Write-Host "Recall:    $($bestModel.recall)" -ForegroundColor White
Write-Host "F1 Score:  $($bestModel.f1_score)" -ForegroundColor White
Write-Host "AUC-ROC:   $($bestModel.auc_roc)" -ForegroundColor White

# Model path mapping
$modelPaths = @{
    "Isolation Forest" = "models/isolation_forest"
    "XGBoost" = "models/xgboost"
    "CatBoost" = "models/catboost"
}

$selectedPath = $modelPaths[$bestModel.model]

Write-Host "`n[3] Model Location:" -ForegroundColor Yellow
Write-Host "Path: $selectedPath" -ForegroundColor White
Write-Host ""
Get-ChildItem $selectedPath -File | Format-Table Name, Length

Write-Host "`n[4] Deployment Readiness Check:" -ForegroundColor Yellow

# Check required files
$requiredFiles = @("model.pkl", "scaler.pkl", "feature_names.txt")
$allPresent = $true

foreach ($file in $requiredFiles) {
    $exists = Test-Path "$selectedPath/$file"
    $status = if ($exists) { "[OK]" } else { "[MISSING]"; $allPresent = $false }
    Write-Host "$status $file" -ForegroundColor $(if ($exists) { "Green" } else { "Red" })
}

Write-Host "`n[5] Next Steps:" -ForegroundColor Yellow
if ($allPresent) {
    Write-Host "[READY] Model is ready for deployment!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To deploy to Triton:" -ForegroundColor Cyan
    Write-Host "  1. Convert to ONNX format" -ForegroundColor White
    Write-Host "  2. Create Triton model repository" -ForegroundColor White
    Write-Host "  3. Deploy to Kubernetes" -ForegroundColor White
    Write-Host ""
    Write-Host "Run: .\scripts\prepare_for_triton.ps1 -Model '$($bestModel.model)'" -ForegroundColor Yellow
} else {
    Write-Host "[ERROR] Model files missing. Re-run training." -ForegroundColor Red
}

Write-Host "`n[6] Performance vs Production Requirements:" -ForegroundColor Yellow
Write-Host ""
Write-Host "Metric         Current  Required  Status" -ForegroundColor Cyan
Write-Host "------         -------  --------  ------" -ForegroundColor Cyan

$requirements = @{
    "Accuracy" = 0.85
    "F1 Score" = 0.80
    "Precision" = 0.85
    "Recall" = 0.75
}

foreach ($metric in $requirements.Keys) {
    $metricKey = $metric.ToLower().Replace(" ", "_")
    $current = $bestModel.$metricKey
    $required = $requirements[$metric]
    $status = if ($current -ge $required) { "PASS" } else { "FAIL" }
    $color = if ($current -ge $required) { "Green" } else { "Red" }
    
    Write-Host ("{0,-14} {1:F4}   {2:F2}      {3}" -f $metric, $current, $required, $status) -ForegroundColor $color
}

Write-Host "`n"
Write-Host "="*80 -ForegroundColor Cyan

# Save selection to file
$selection = @{
    selected_model = $bestModel.model
    selection_criterion = $Criterion
    model_path = $selectedPath
    metrics = $bestModel
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    ready_for_deployment = $allPresent
}

$selection | ConvertTo-Json -Depth 10 | Out-File "model_selection.json"
Write-Host "Selection saved to: model_selection.json" -ForegroundColor Green