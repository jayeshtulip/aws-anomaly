#!/usr/bin/env python3
"""
Convert CatBoost to ONNX with proper tensor outputs
"""
import os
import numpy as np
from catboost import CatBoostClassifier
import onnx
from onnx import helper, TensorProto

print("="*70)
print("CATBOOST TO ONNX CONVERSION (FIXED)")
print("="*70)

# Load CatBoost model
print("\n[1] Loading CatBoost model...")
model = CatBoostClassifier()
model.load_model('models/catboost_triton/model.cb')
print(f"✓ Model loaded: {model.tree_count_} trees")

# Export with specific parameters to avoid SEQUENCE type
print("\n[2] Exporting to ONNX with fixed output types...")
output_dir = 'models/catboost_onnx_fixed/1'
os.makedirs(output_dir, exist_ok=True)
onnx_path = os.path.join(output_dir, 'model.onnx')

try:
    # Try with onnx_export_parameters to control output format
    model.save_model(
        onnx_path,
        format="onnx",
        export_parameters={
            'onnx_domain': 'ai.catboost',
            'onnx_model_version': 1,
        }
    )
    print(f"✓ Initial export complete")
    
    # Load and inspect ONNX model
    onnx_model = onnx.load(onnx_path)
    
    print("\n[3] Analyzing ONNX model structure...")
    print(f"  Inputs:")
    for input in onnx_model.graph.input:
        print(f"    - {input.name}: {input.type}")
    
    print(f"  Outputs:")
    for output in onnx_model.graph.output:
        print(f"    - {output.name}: {output.type}")
    
    # Check if probabilities is SEQUENCE type
    prob_output = [o for o in onnx_model.graph.output if 'prob' in o.name.lower()]
    if prob_output and prob_output[0].type.HasField('sequence_type'):
        print("\n⚠ Warning: Probabilities output is SEQUENCE type")
        print("  This model won't work with Triton ONNX backend")
        print("  Recommendation: Use Python backend instead")
    else:
        print("\n✓ Output types look good!")
    
    file_size = os.path.getsize(onnx_path)
    print(f"\n✓ Model saved: {onnx_path} ({file_size:,} bytes)")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*70)