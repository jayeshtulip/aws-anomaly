"""Configuration loader utility."""
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage configuration files."""

    def __init__(self, config_dir: str = "config"):
        """Initialize config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        load_dotenv()  # Load environment variables from .env

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Args:
            config_file: Name of config file (e.g., 'config.yaml')
            
        Returns:
            Dictionary containing configuration
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Replace environment variables
        config = self._replace_env_vars(config)
        
        return config

    def _replace_env_vars(self, config: Any) -> Any:
        """Recursively replace ${VAR} with environment variable values.
        
        Args:
            config: Configuration dictionary or value
            
        Returns:
            Configuration with environment variables replaced
        """
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config

    def get_aws_config(self) -> Dict[str, str]:
        """Get AWS configuration from environment.
        
        Returns:
            Dictionary with AWS configuration
        """
        return {
            'account_id': os.getenv('AWS_ACCOUNT_ID'),
            'region': os.getenv('AWS_REGION', 'us-east-1'),
            's3_bucket': os.getenv('S3_BUCKET_NAME'),
            'eks_cluster': os.getenv('EKS_CLUSTER_NAME')
        }

    def get_mlflow_config(self) -> Dict[str, str]:
        """Get MLflow configuration from environment.
        
        Returns:
            Dictionary with MLflow configuration
        """
        return {
            'tracking_uri': os.getenv('MLFLOW_TRACKING_URI'),
            'artifact_root': os.getenv('MLFLOW_ARTIFACT_ROOT'),
            's3_endpoint': os.getenv('MLFLOW_S3_ENDPOINT_URL')
        }


# Global config loader instance
config_loader = ConfigLoader()