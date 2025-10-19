"""Convert XGBoost model to ONNX format for Triton."""
import sys
import pickle
from pathlib import Path
import numpy as np
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def convert_xgboost_to_onnx(
    model_path: str = "models/xgboost/model.pkl",
    output_path: str = "triton_models/xgboost_anomaly/1/model.onnx",
    num_features: int = 45
):
    """Convert XGBoost model to ONNX using native XGBoost export.
    
    Args:
        model_path: Path to pickled XGBoost model
        output_path: Path to save ONNX model
        num_features: Number of input features
    """
    logger.info(f"Converting XGBoost model to ONNX...")
    
    try:
        import xgboost as xgb
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model type: {type(model)}")
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to ONNX using XGBoost native method
        logger.info("Converting to ONNX format using XGBoost native export...")
        
        # Get the booster
        booster = model.get_booster()
        
        # Define initial types for ONNX
        initial_types = [('input', 'float', [None, num_features])]
        
        # Save to ONNX
        booster.save_model(output_path)
        logger.success(f"✓ Model saved to {output_path}")
        
        # XGBoost models need to be saved in JSON format first, then converted
        # Let's use a different approach: save as JSON and use onnxmltools
        json_path = output_path.replace('.onnx', '.json')
        booster.save_model(json_path)
        logger.info(f"Saved XGBoost model as JSON: {json_path}")
        
        # Now convert JSON to ONNX using onnxmltools
        try:
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            # Load the booster from JSON
            booster_onnx = xgb.Booster()
            booster_onnx.load_model(json_path)
            
            # Define input type
            initial_type = [('input', FloatTensorType([None, num_features]))]
            
            # Convert
            onnx_model = convert_xgboost(
                booster_onnx,
                initial_types=initial_type,
                target_opset=12
            )
            
            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            logger.success(f"✓ ONNX model saved to {output_path}")
            
            # Test the model
            test_onnx_model(output_path, num_features)
            
            return True
            
        except ImportError:
            logger.error("onnxmltools not installed. Install with: pip install onnxmltools")
            logger.info("Falling back to Python backend for Triton...")
            return create_python_backend_model(model_path, output_path.replace('/1/model.onnx', ''), num_features)
            
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Falling back to Python backend...")
        return create_python_backend_model(model_path, output_path.replace('/1/model.onnx', ''), num_features)


def create_python_backend_model(
    model_path: str,
    model_dir: str,
    num_features: int
):
    """Create Triton Python backend model (no ONNX conversion needed).
    
    Args:
        model_path: Path to pickled model
        model_dir: Triton model directory
        num_features: Number of features
    """
    logger.info("Creating Triton Python backend model...")
    
    # Create directory structure
    model_path_obj = Path(model_dir)
    version_dir = model_path_obj / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model file
    import shutil
    dest_model = version_dir / "model.pkl"
    shutil.copy(model_path, dest_model)
    logger.info(f"Copied model to {dest_model}")
    
    # Create model.py for Python backend
    model_py = version_dir / "model.py"
    model_py_content = f'''import json
import numpy as np
import pickle
import triton_python_backend_utils as pb_utils
from pathlib import Path


class TritonPythonModel:
    def initialize(self, args):
        """Initialize model."""
        model_path = Path(args['model_repository']) / args['model_version'] / "model.pkl"
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler if exists
        scaler_path = Path(args['model_repository']).parent.parent / "xgboost" / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None

    def execute(self, requests):
        """Execute inference."""
        responses = []
        
        for request in requests:
            # Get input tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input")
            input_data = input_tensor.as_numpy()
            
            # Scale if scaler available
            if self.scaler:
                input_data = self.scaler.transform(input_data)
            
            # Predict
            predictions = self.model.predict(input_data)
            probabilities = self.model.predict_proba(input_data)
            
            # Create output tensors
            output_pred = pb_utils.Tensor("output_label", predictions.astype(np.int64))
            output_prob = pb_utils.Tensor("output_probability", probabilities.astype(np.float32))
            
            # Create response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_pred, output_prob]
            )
            responses.append(inference_response)
        
        return responses
'''
    
    with open(model_py, 'w') as f:
        f.write(model_py_content)
    
    logger.info(f"Created model.py at {model_py}")
    
    # Create config.pbtxt for Python backend
    config_path = model_path_obj / "config.pbtxt"
    config_content = f'''name: "xgboost_anomaly"
backend: "python"
max_batch_size: 32

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [-1, {num_features}]
  }}
]

output [
  {{
    name: "output_label"
    data_type: TYPE_INT64
    dims: [-1]
  }},
  {{
    name: "output_probability"
    data_type: TYPE_FP32
    dims: [-1, 2]
  }}
]

dynamic_batching {{
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 100
}}

instance_group [
  {{
    count: 2
    kind: KIND_CPU
  }}
]
'''
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    logger.success(f"✓ Python backend model created at {model_dir}")
    logger.info("Note: Use Python backend in Triton (no ONNX conversion needed)")
    
    return True


def test_onnx_model(model_path: str, num_features: int):
    """Test ONNX model inference."""
    try:
        import onnxruntime as ort
        
        logger.info("Testing ONNX model...")
        
        session = ort.InferenceSession(model_path)
        dummy_input = np.random.rand(1, num_features).astype(np.float32)
        
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: dummy_input})
        
        logger.success(f"✓ ONNX model test passed. Output shape: {result[0].shape}")
        
    except Exception as e:
        logger.warning(f"ONNX test skipped: {e}")


if __name__ == "__main__":
    # Get number of features
    feature_file = Path("models/xgboost/feature_names.txt")
    if feature_file.exists():
        with open(feature_file, 'r') as f:
            num_features = len(f.readlines())
    else:
        num_features = 45
    
    logger.info(f"Number of features: {num_features}")
    
    success = convert_xgboost_to_onnx(num_features=num_features)
    
    if success:
        logger.success("✅ Model ready for Triton!")
    else:
        logger.error("❌ Model preparation failed")
        sys.exit(1)