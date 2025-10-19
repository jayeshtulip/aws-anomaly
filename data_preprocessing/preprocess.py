"""Main preprocessing pipeline."""
import os
import sys
from pathlib import Path
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_preprocessing.log_parser import LogParser
from data_preprocessing.feature_engineering import FeatureEngineer
from data_preprocessing.data_validator import DataValidator
from utils.config_loader import config_loader
from utils.aws_utils import S3Handler


def load_raw_logs(source: str) -> pd.DataFrame:
    """Load raw logs from source.
    
    Args:
        source: Path to raw logs (local or S3)
        
    Returns:
        DataFrame with raw log data
    """
    logger.info(f"Loading raw logs from {source}")
    
    if source.startswith('s3://'):
        # Load from S3
        s3_handler = S3Handler()
        df = s3_handler.read_csv_from_s3(source)
    else:
        # Load from local file
        df = pd.read_csv(source)
    
    logger.info(f"Loaded {len(df)} raw log entries")
    return df


def preprocess_pipeline(
    input_path: str,
    output_path: str,
    config_path: str = "config/config.yaml"
) -> pd.DataFrame:
    """Run complete preprocessing pipeline.
    
    Args:
        input_path: Path to raw log data
        output_path: Path to save processed features
        config_path: Path to configuration file
        
    Returns:
        Processed DataFrame
    """
    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 60)
    
    # Load configuration
    config = config_loader.load_config('config.yaml')
    
    # Step 1: Load raw logs
    logger.info("\n[Step 1/5] Loading raw logs...")
    df_raw = load_raw_logs(input_path)
    
    # Step 2: Parse logs
    logger.info("\n[Step 2/5] Parsing logs...")
    parser = LogParser()
    
    # If df_raw has log events format, parse them
    if 'message' in df_raw.columns:
        log_events = df_raw.to_dict('records')
        df_parsed = parser.parse_logs_to_dataframe(log_events)
    else:
        df_parsed = df_raw
    
    # Step 3: Engineer features
    logger.info("\n[Step 3/5] Engineering features...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df_parsed)
    
    # Step 4: Validate data
    logger.info("\n[Step 4/5] Validating data quality...")
    validator = DataValidator()
    
    validation_results = validator.validate_all(
        df_features,
        critical_columns=['timestamp', 'log_level'],
        categorical_columns={
            'log_level': ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL']
        }
    )
    
    if not validation_results['is_valid']:
        logger.warning("Data validation found issues (continuing anyway)")
    
    # Step 5: Save processed data
    logger.info("\n[Step 5/5] Saving processed features...")
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df_features.to_csv(output_path, index=False)
    logger.success(f"Saved processed features to {output_path}")
    
    # Also save to S3 if configured
    aws_config = config_loader.get_aws_config()
    if aws_config.get('s3_bucket'):
        s3_handler = S3Handler()
        s3_path = f"s3://{aws_config['s3_bucket']}/data/processed/features.csv"
        s3_handler.upload_file(output_path, s3_path)
        logger.info(f"Uploaded to S3: {s3_path}")
    
    logger.info("=" * 60)
    logger.success(f"Preprocessing complete! Shape: {df_features.shape}")
    logger.info("=" * 60)
    
    return df_features


if __name__ == "__main__":
    load_dotenv()
    
    # Configure paths
    input_path = "data/raw/cloudwatch_logs.csv"
    output_path = "data/processed/features.csv"
    
    # Run pipeline
    df_processed = preprocess_pipeline(input_path, output_path)
    
    # Show summary
    logger.info("\nProcessed Features Summary:")
    logger.info(f"Shape: {df_processed.shape}")
    logger.info(f"Columns: {list(df_processed.columns[:10])}...")
    logger.info(f"Memory usage: {df_processed.memory_usage(deep=True).sum() / 1024**2:.2f} MB")