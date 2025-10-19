"""Feature engineering for anomaly detection."""
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
from loguru import logger


class FeatureEngineer:
    """Engineer features from parsed log data."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.numeric_features = []
        self.categorical_features = []
        self.time_features = []
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features.
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            DataFrame with added time features
        """
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found")
            return df
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Business hours (9 AM - 5 PM, Monday-Friday)
        df['is_business_hours'] = (
            (df['hour'] >= 9) & 
            (df['hour'] < 17) & 
            (df['day_of_week'] < 5)
        ).astype(int)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        logger.info("Added time-based features")
        self.time_features = [
            'hour', 'day_of_week', 'day_of_month', 'month', 
            'is_weekend', 'is_business_hours'
        ]
        
        return df
    
    def add_log_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log level features.
        
        Args:
            df: DataFrame with 'log_level' column
            
        Returns:
            DataFrame with encoded log levels
        """
        if 'log_level' not in df.columns:
            logger.warning("No log_level column found")
            return df
        
        df = df.copy()
        
        # One-hot encode log level
        log_level_dummies = pd.get_dummies(
            df['log_level'], 
            prefix='log_level',
            drop_first=False
        )
        
        df = pd.concat([df, log_level_dummies], axis=1)
        
        # Log level severity score
        severity_map = {
            'DEBUG': 1,
            'INFO': 2,
            'WARN': 3,
            'ERROR': 4,
            'FATAL': 5
        }
        df['log_severity'] = df['log_level'].map(severity_map).fillna(2)
        
        logger.info("Added log level features")
        return df
    
    def add_error_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add error-related features.
        
        Args:
            df: DataFrame with error columns
            
        Returns:
            DataFrame with error features
        """
        df = df.copy()
        
        # Binary error indicator
        df['has_error'] = (
            (df['log_level'] == 'ERROR') | 
            (df['error_type'].notna())
        ).astype(int)
        
        # Error type encoding
        if 'error_type' in df.columns:
            df['error_type_encoded'] = pd.Categorical(
                df['error_type'].fillna('None')
            ).codes
        
        # Error message length
        if 'error_message' in df.columns:
            df['error_message_length'] = df['error_message'].fillna('').str.len()
        
        logger.info("Added error features")
        return df
    
    def add_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add performance-related features.
        
        Args:
            df: DataFrame with performance metrics
            
        Returns:
            DataFrame with performance features
        """
        df = df.copy()
        
        # Duration features
        if 'duration' in df.columns:
            df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
            df['duration_log'] = np.log1p(df['duration'].fillna(0))
            
            # Flag slow requests (> 3 seconds)
            df['is_slow'] = (df['duration'] > 3000).astype(int)
        
        # Memory features
        if 'memory_used' in df.columns:
            df['memory_used'] = pd.to_numeric(df['memory_used'], errors='coerce')
            df['memory_used_mb'] = df['memory_used'] / (1024 * 1024)
        
        logger.info("Added performance features")
        return df
    
    def add_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add service-related features.
        
        Args:
            df: DataFrame with 'service_name' column
            
        Returns:
            DataFrame with service features
        """
        if 'service_name' not in df.columns:
            logger.warning("No service_name column found")
            return df
        
        df = df.copy()
        
        # Service name encoding
        df['service_encoded'] = pd.Categorical(df['service_name']).codes
        
        # One-hot encode top services
        top_services = df['service_name'].value_counts().head(10).index
        for service in top_services:
            df[f'service_{service}'] = (df['service_name'] == service).astype(int)
        
        logger.info("Added service features")
        return df
    
    def add_aggregation_features(
        self,
        df: pd.DataFrame,
        window: str = '5min'
    ) -> pd.DataFrame:
        """Add rolling aggregation features.
        
        Args:
            df: DataFrame with timestamp index
            window: Rolling window size
            
        Returns:
            DataFrame with aggregation features
        """
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found")
            return df
        
        df = df.copy()
        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')
        
        # Error rate in window
        if 'has_error' in df.columns:
            df['error_rate'] = df['has_error'].rolling(window).mean()
            df['error_count'] = df['has_error'].rolling(window).sum()
        
        # Request count - use count instead of size
        df['request_count'] = df['log_level'].rolling(window).count()
        
        # Average duration in window
        if 'duration' in df.columns:
            df['avg_duration'] = df['duration'].rolling(window).mean()
            df['max_duration'] = df['duration'].rolling(window).max()
        
        df = df.reset_index()
        
        logger.info(f"Added rolling aggregation features (window={window})")
        return df
    
    def add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add text-based features from log messages.
        
        Args:
            df: DataFrame with 'message' column
            
        Returns:
            DataFrame with text features
        """
        if 'message' not in df.columns:
            logger.warning("No message column found")
            return df
        
        df = df.copy()
        
        # Message length
        df['message_length'] = df['message'].str.len()
        
        # Word count
        df['word_count'] = df['message'].str.split().str.len()
        
        # Contains keywords
        keywords = {
            'timeout': r'timeout|timed out',
            'connection': r'connection|connect',
            'auth': r'auth|authorization|unauthorized',
            'permission': r'permission|denied|forbidden',
            'memory': r'memory|oom|out of memory',
            'cpu': r'cpu|processor',
            'disk': r'disk|storage|space',
            'network': r'network|dns|socket'
        }
        
        for keyword, pattern in keywords.items():
            df[f'has_{keyword}'] = df['message'].str.contains(
                pattern, 
                case=False, 
                na=False
            ).astype(int)
        
        logger.info("Added text features")
        return df
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        aggregation_window: str = '5min'
    ) -> pd.DataFrame:
        """Create all features from raw log data.
        
        Args:
            df: Raw parsed log DataFrame
            aggregation_window: Window for rolling features
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        # Add all feature types
        df = self.add_time_features(df)
        df = self.add_log_level_features(df)
        df = self.add_error_features(df)
        df = self.add_performance_features(df)
        df = self.add_service_features(df)
        df = self.add_text_features(df)
        df = self.add_aggregation_features(df, aggregation_window)
        
        # Fill NaN values
        df = df.fillna({
            'duration': 0,
            'memory_used': 0,
            'error_rate': 0,
            'error_count': 0,
            'request_count': 0,
            'avg_duration': 0,
            'max_duration': 0
        })
        
        logger.success(f"Feature engineering complete. Shape: {df.shape}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all created feature names.
        
        Returns:
            List of feature names
        """
        return self.numeric_features + self.categorical_features + self.time_features


if __name__ == "__main__":
    # Example usage
    from datetime import timedelta
    
    # Create sample data
    n_samples = 1000
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n_samples)]
    
    sample_df = pd.DataFrame({
        'timestamp': timestamps,
        'message': ['Sample log message'] * n_samples,
        'log_level': np.random.choice(['INFO', 'WARN', 'ERROR'], n_samples),
        'service_name': np.random.choice(['api', 'worker', 'scheduler'], n_samples),
        'duration': np.random.randint(100, 5000, n_samples),
        'memory_used': np.random.randint(1000000, 10000000, n_samples),
        'error_type': [None] * (n_samples - 10) + ['TimeoutError'] * 10
    })
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(sample_df)
    
    logger.info(f"Features created: {list(features_df.columns)}")
    logger.info(f"Shape: {features_df.shape}")