# Prepare Model for Triton Deployment
param(
    [string]$ModelSelection = "model_selection.json"
)

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "PREPARING MODEL FOR TRITON DEPLOYMENT" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan

# Load model selection
if (Test-Path $ModelSelection) {
    $selection = Get-Content $ModelSelection | ConvertFrom-Json
    $model = $selection.selected_model
    $modelPath = $selection.model_path
    
    Write-Host "`nSelected Model: $model" -ForegroundColor Green
    Write-Host "Model Path: $modelPath" -ForegroundColor White
} else {
    Write-Host "ERROR: model_selection.json not found. Run .\scripts\select_best_model.ps1 first" -ForegroundColor Red
    exit 1
}

# Create Triton model repository structure
Write-Host "`n[1/5] Creating Triton repository structure..." -ForegroundColor Yellow
$tritonPath = "triton_models/anomaly_detector"
New-Item -ItemType Directory -Force -Path "$tritonPath/1" | Out-Null

# Copy model artifacts
Write-Host "[2/5] Copying model artifacts..." -ForegroundColor Yellow
Copy-Item "$modelPath/model.pkl" -Destination "$tritonPath/1/" -Force
Copy-Item "$modelPath/scaler.pkl" -Destination "$tritonPath/1/" -Force
Copy-Item "$modelPath/feature_names.txt" -Destination "$tritonPath/1/" -Force

Write-Host "  ✓ model.pkl" -ForegroundColor Green
Write-Host "  ✓ scaler.pkl" -ForegroundColor Green
Write-Host "  ✓ feature_names.txt" -ForegroundColor Green

# Create model.py for Python backend
Write-Host "[3/5] Creating Triton Python backend..." -ForegroundColor Yellow

# Use a proper here-string with single quotes to avoid variable expansion
@'
import json
import numpy as np
import pickle
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        """Initialize model"""
        self.model_config = json.loads(args['model_config'])
        
        # Load model
        model_path = '/models/anomaly_detector/1/model.pkl'
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        scaler_path = '/models/anomaly_detector/1/scaler.pkl'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature names
        feature_names_path = '/models/anomaly_detector/1/feature_names.txt'
        with open(feature_names_path, 'r') as f:
            self.feature_names = [line.strip() for line in f]
        
        print(f"Model loaded: {type(self.model).__name__}")
        print(f"Features: {len(self.feature_names)}")

    def execute(self, requests):
        """Execute inference requests"""
        responses = []
        
        for request in requests:
            # Get input tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_data = input_tensor.as_numpy()
            
            # Scale features
            scaled_data = self.scaler.transform(input_data)
            
            # Predict
            predictions = self.model.predict(scaled_data)
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(scaled_data)[:, 1]
            else:
                probas = predictions.astype(np.float32)
            
            # Handle Isolation Forest output (-1, 1) -> (0, 1)
            if hasattr(self.model, 'decision_function'):
                predictions = (predictions == -1).astype(np.int32)
            
            # Create output tensors
            pred_tensor = pb_utils.Tensor("PREDICTION", predictions.astype(np.int32))
            proba_tensor = pb_utils.Tensor("PROBABILITY", probas.astype(np.float32))
            
            response = pb_utils.InferenceResponse(output_tensors=[pred_tensor, proba_tensor])
            responses.append(response)
        
        return responses

    def finalize(self):
        """Cleanup"""
        print('Model finalized')
'@ | Out-File "$tritonPath/1/model.py" -Encoding UTF8

Write-Host "  ✓ model.py created" -ForegroundColor Green

# Create config.pbtxt
Write-Host "[4/5] Creating Triton config..." -ForegroundColor Yellow

@'
name: "anomaly_detector"
backend: "python"
max_batch_size: 64

input [
  {
    name: "INPUT"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]

output [
  {
    name: "PREDICTION"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  {
    name: "PROBABILITY"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 100
}
'@ | Out-File "$tritonPath/config.pbtxt" -Encoding ASCII

Write-Host "  ✓ config.pbtxt created" -ForegroundColor Green

# Create metadata
Write-Host "[5/5] Creating metadata..." -ForegroundColor Yellow

$metadata = @{
    model_name = "anomaly_detector"
    model_type = $model
    version = "1"
    created_at = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    metrics = $selection.metrics
    deployment_ready = $true
}

$metadata | ConvertTo-Json -Depth 10 | Out-File "$tritonPath/metadata.json"
Write-Host "  ✓ metadata.json created" -ForegroundColor Green

# Summary
Write-Host "`n" + ("="*80) -ForegroundColor Cyan
Write-Host "MODEL PREPARED FOR TRITON" -ForegroundColor Green
Write-Host ("="*80) -ForegroundColor Cyan

Write-Host "`nTriton Model Repository:" -ForegroundColor Yellow
Get-ChildItem $tritonPath -Recurse -File | Select-Object FullName, Length | Format-Table -AutoSize

Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "1. Upload to S3:" -ForegroundColor Cyan
Write-Host "   aws s3 sync triton_models/ s3://triton-models-71544/models/" -ForegroundColor White
Write-Host ""
Write-Host "2. Deploy to Kubernetes:" -ForegroundColor Cyan
Write-Host "   kubectl apply -f k8s/triton-deployment.yaml" -ForegroundColor White
Write-Host ""
Write-Host "3. Verify deployment:" -ForegroundColor Cyan
Write-Host "   kubectl get pods -n ml-inference" -ForegroundColor White

Write-Host "`n" + ("="*80) -ForegroundColor Cyan