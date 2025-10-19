#!/usr/bin/env python3
"""
Convert CatBoost model to ONNX format for Triton Inference Server
"""
import os
import numpy as np
from catboost import CatBoostClassifier

print("="*70)
print("CATBOOST TO ONNX CONVERSION")
print("="*70)

# Load CatBoost model
print("\n[1] Loading CatBoost model...")
model = CatBoostClassifier()
model.load_model('models/catboost_triton/model.cb')

print(f"✓ Model loaded successfully")
print(f"  Tree count: {model.tree_count_}")
print(f"  Feature names: {len(model.feature_names_) if model.feature_names_ else 'N/A'}")

# Create output directory
output_dir = 'models/catboost_onnx/1'
os.makedirs(output_dir, exist_ok=True)

# Export to ONNX
print("\n[2] Converting to ONNX format...")
onnx_path = os.path.join(output_dir, 'model.onnx')

model.save_model(
    onnx_path,
    format="onnx",
    export_parameters={
        'onnx_domain': 'ai.catboost',
        'onnx_model_version': 1,
        'onnx_doc_string': 'CatBoost Anomaly Detection Model',
        'onnx_graph_name': 'CatBoostModel'
    }
)

print(f"✓ ONNX model saved to: {onnx_path}")

# Verify file size
file_size = os.path.getsize(onnx_path)
print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

print("\n" + "="*70)
print("✅ CONVERSION COMPLETE!")
print("="*70)