import json

# Check XGBoost model
with open('models/xgboost_anomaly/1/model.json', 'r') as f:
    xgb_model = json.load(f)
    
print("XGBoost feature info:")
if 'learner' in xgb_model and 'feature_names' in xgb_model['learner']:
    features = xgb_model['learner']['feature_names']
    print(f"  Feature count: {len(features)}")
    print(f"  Features: {features[:5]}...")  # First 5
else:
    print("  Feature names not found in model")

# Check processed data
import pandas as pd
try:
    train_data = pd.read_csv('data/processed/train_features.csv')
    print(f"\nTraining data features: {train_data.shape[1]}")
    print(f"Feature names: {list(train_data.columns)[:5]}...")
except:
    print("\nTraining data not found")