"""Generate sample CloudWatch log data for testing."""
import json
import random
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from loguru import logger


class SampleDataGenerator:
    """Generate sample log data."""
    
    def __init__(self):
        """Initialize generator."""
        self.services = ['api-gateway', 'lambda-processor', 'data-pipeline', 'auth-service']
        self.log_levels = ['INFO', 'WARN', 'ERROR', 'DEBUG']
        self.error_types = [
            'TimeoutError', 
            'ConnectionError', 
            'AuthenticationError',
            'ValidationError',
            'DatabaseError',
            None
        ]
        
        self.messages = [
            "Request processed successfully",
            "User authentication completed",
            "Database query executed",
            "Cache miss, fetching from database",
            "API rate limit warning",
            "Connection timeout to external service",
            "Invalid input parameters received",
            "Memory usage above threshold",
            "Slow query detected",
            "Service health check passed"
        ]
    
    def generate_log_entry(
        self,
        timestamp: datetime,
        inject_anomaly: bool = False
    ) -> dict:
        """Generate a single log entry.
        
        Args:
            timestamp: Timestamp for the log
            inject_anomaly: Whether to inject an anomaly
            
        Returns:
            Log entry dictionary
        """
        # Select log level (more errors if anomaly)
        if inject_anomaly:
            log_level = random.choice(['ERROR', 'ERROR', 'WARN'])
        else:
            log_level = random.choice(
                ['INFO'] * 7 + ['WARN'] * 2 + ['ERROR'] * 1
            )
        
        # Generate duration (slower if anomaly)
        if inject_anomaly:
            duration = random.randint(3000, 10000)  # 3-10 seconds
        else:
            duration = random.randint(50, 2000)  # 50ms - 2s
        
        # Select error type
        error_type = None
        error_message = None
        if log_level == 'ERROR':
            error_type = random.choice([et for et in self.error_types if et])
            error_message = f"{error_type}: Operation failed"
        
        # Create log entry
        entry = {
            'timestamp': timestamp,
            'log_stream': f"/aws/{random.choice(self.services)}/instance-{random.randint(1, 5)}",
            'message': random.choice(self.messages),
            'log_level': log_level,
            'service_name': random.choice(self.services),
            'duration': duration,
            'memory_used': random.randint(50000000, 200000000),  # 50-200 MB
            'request_id': f"req-{random.randint(100000, 999999)}",
            'error_type': error_type,
            'error_message': error_message
        }
        
        return entry
    
    def generate_dataset(
        self,
        start_time: datetime,
        duration_hours: int = 24,
        events_per_hour: int = 100,
        anomaly_percentage: float = 0.1
    ) -> pd.DataFrame:
        """Generate a dataset of log entries.
        
        Args:
            start_time: Start time for logs
            duration_hours: Duration in hours
            events_per_hour: Number of events per hour
            anomaly_percentage: Percentage of anomalies (0.0 - 1.0)
            
        Returns:
            DataFrame with log entries
        """
        total_events = duration_hours * events_per_hour
        num_anomalies = int(total_events * anomaly_percentage)
        
        logger.info(f"Generating {total_events} log events ({num_anomalies} anomalies)")
        
        entries = []
        
        for i in range(total_events):
            # Calculate timestamp
            minutes_offset = (i / events_per_hour) * 60
            timestamp = start_time + timedelta(minutes=minutes_offset)
            
            # Decide if this should be an anomaly
            inject_anomaly = i < num_anomalies and random.random() < 0.3
            
            entry = self.generate_log_entry(timestamp, inject_anomaly)
            entries.append(entry)
        
        df = pd.DataFrame(entries)
        
        # Shuffle to distribute anomalies
        df = df.sample(frac=1).reset_index(drop=True)
        
        logger.success(f"Generated {len(df)} log entries")
        
        return df
    
    def generate_and_save(
        self,
        output_path: str = "data/raw/cloudwatch_logs.csv",
        duration_hours: int = 24,
        events_per_hour: int = 100
    ):
        """Generate dataset and save to file.
        
        Args:
            output_path: Path to save the dataset
            duration_hours: Duration in hours
            events_per_hour: Events per hour
        """
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate data
        start_time = datetime.now() - timedelta(hours=duration_hours)
        df = self.generate_dataset(start_time, duration_hours, events_per_hour)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.success(f"Saved {len(df)} log entries to {output_path}")
        
        # Print summary
        logger.info("\nDataset Summary:")
        logger.info(f"  Total entries: {len(df)}")
        logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"  Log level distribution:\n{df['log_level'].value_counts()}")
        logger.info(f"  Services: {df['service_name'].unique().tolist()}")


if __name__ == "__main__":
    generator = SampleDataGenerator()
    
    # Generate 24 hours of data with 100 events per hour
    generator.generate_and_save(
        output_path="data/raw/cloudwatch_logs.csv",
        duration_hours=24,
        events_per_hour=100
    )