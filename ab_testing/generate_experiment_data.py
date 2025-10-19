"""Generate synthetic A/B test experiment data."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger


def generate_experiment_data(
    n_samples_per_variant: int = 1000,
    output_file: str = "ab_testing/experiment_data.csv"
):
    """Generate synthetic A/B test data."""
    logger.info(f"Generating experiment data with {n_samples_per_variant} samples per variant...")
    
    np.random.seed(42)
    
    variants = ['xgb_optimized_v1', 'xgb_optimized_v2', 'xgb_optimized_v3']
    
    data = []
    base_time = datetime.now() - timedelta(days=7)
    
    for variant_id in variants:
        # Add variant-specific performance characteristics
        if variant_id == 'xgb_optimized_v1':
            f1_mean, f1_std = 0.95, 0.03
            latency_mean, latency_std = 45, 8
            accuracy_mean = 0.94
        elif variant_id == 'xgb_optimized_v2':
            f1_mean, f1_std = 0.96, 0.025
            latency_mean, latency_std = 52, 10
            accuracy_mean = 0.95
        else:  # v3
            f1_mean, f1_std = 0.94, 0.035
            latency_mean, latency_std = 38, 7
            accuracy_mean = 0.93
        
        for i in range(n_samples_per_variant):
            timestamp = base_time + timedelta(minutes=i*10)
            
            record = {
                'timestamp': timestamp.isoformat(),
                'variant_id': variant_id,
                'f1_score': np.clip(np.random.normal(f1_mean, f1_std), 0, 1),
                'accuracy': np.clip(np.random.normal(accuracy_mean, 0.03), 0, 1),
                'precision': np.clip(np.random.normal(0.93, 0.04), 0, 1),
                'recall': np.clip(np.random.normal(0.92, 0.04), 0, 1),
                'latency_ms': np.clip(np.random.normal(latency_mean, latency_std), 20, 100),
                'prediction': np.random.choice([0, 1], p=[0.9, 0.1]),
                'confidence': np.random.uniform(0.7, 1.0),
                'error_count': np.random.poisson(2)
            }
            
            data.append(record)
    
    df = pd.DataFrame(data)
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.success(f"✅ Generated {len(df)} samples, saved to {output_path}")
    return df


if __name__ == "__main__":
    df = generate_experiment_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nVariants: {df['variant_id'].value_counts()}")