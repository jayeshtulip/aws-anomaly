# Pipeline Results Summary
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "DVC PIPELINE RESULTS" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan

Write-Host "`n[1] Pipeline Metrics" -ForegroundColor Yellow
dvc metrics show

Write-Host "`n[2] Pipeline Structure" -ForegroundColor Yellow
dvc dag

Write-Host "`n[3] Generated Files" -ForegroundColor Yellow
Write-Host "`nModels:" -ForegroundColor Cyan
Get-ChildItem models -Recurse -Include *.pkl | Select-Object FullName, Length

Write-Host "`nMetrics:" -ForegroundColor Cyan
Get-ChildItem metrics -Include *.json | Select-Object Name, Length

Write-Host "`nPlots:" -ForegroundColor Cyan
Get-ChildItem plots -Include *.png | Select-Object Name, Length

Write-Host "`n[4] Best Model" -ForegroundColor Yellow
$comparison = Get-Content metrics/model_comparison.json | ConvertFrom-Json
$best = $comparison | Sort-Object f1_score -Descending | Select-Object -First 1
Write-Host "Model: $($best.model)" -ForegroundColor Green
Write-Host "F1 Score: $($best.f1_score)" -ForegroundColor Green
Write-Host "Accuracy: $($best.accuracy)" -ForegroundColor Green
Write-Host "AUC-ROC: $($best.auc_roc)" -ForegroundColor Green

Write-Host "`n[5] Storage Locations" -ForegroundColor Yellow
Write-Host "S3: s3://triton-models-71544/dvc-storage/" -ForegroundColor White
Write-Host "MLflow: http://localhost:5000" -ForegroundColor White

Write-Host "`n"
Write-Host "="*80 -ForegroundColor Cyan