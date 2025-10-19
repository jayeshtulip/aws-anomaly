"""Parse and clean AWS CloudWatch logs."""
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
from loguru import logger


class LogParser:
    """Parse CloudWatch log entries."""
    
    def __init__(self):
        """Initialize log parser."""
        self.error_patterns = [
            r'error',
            r'exception',
            r'failed',
            r'timeout',
            r'cannot',
            r'unable',
            r'denied'
        ]
        
        self.warning_patterns = [
            r'warning',
            r'warn',
            r'deprecated',
            r'retry'
        ]
    
    def parse_json_log(self, log_message: str) -> Optional[Dict[str, Any]]:
        """Parse JSON formatted log message.
        
        Args:
            log_message: Raw log message
            
        Returns:
            Parsed log dictionary or None
        """
        try:
            return json.loads(log_message)
        except json.JSONDecodeError:
            return None
    
    def extract_log_level(self, message: str) -> str:
        """Extract log level from message.
        
        Args:
            message: Log message
            
        Returns:
            Log level (ERROR, WARN, INFO, DEBUG)
        """
        message_lower = message.lower()
        
        # Check for error patterns
        for pattern in self.error_patterns:
            if re.search(pattern, message_lower):
                return 'ERROR'
        
        # Check for warning patterns
        for pattern in self.warning_patterns:
            if re.search(pattern, message_lower):
                return 'WARN'
        
        # Check explicit level indicators
        if 'info' in message_lower or 'information' in message_lower:
            return 'INFO'
        if 'debug' in message_lower:
            return 'DEBUG'
        
        return 'INFO'  # Default
    
    def extract_service_name(self, log_stream: str) -> str:
        """Extract service name from log stream.
        
        Args:
            log_stream: CloudWatch log stream name
            
        Returns:
            Service name
        """
        # Example: /aws/lambda/function-name -> lambda
        # Example: /ecs/service-name -> ecs
        if not log_stream:
            return 'unknown'
        
        parts = log_stream.split('/')
        if len(parts) >= 3:
            return parts[2]
        return 'unknown'
    
    def parse_timestamp(
        self,
        timestamp: Any,
        format: Optional[str] = None
    ) -> datetime:
        """Parse timestamp from various formats.
        
        Args:
            timestamp: Timestamp value
            format: Optional strptime format
            
        Returns:
            Parsed datetime
        """
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, (int, float)):
            # Unix timestamp (milliseconds or seconds)
            if timestamp > 1e10:  # Milliseconds
                return datetime.fromtimestamp(timestamp / 1000)
            return datetime.fromtimestamp(timestamp)
        
        if isinstance(timestamp, str):
            if format:
                return datetime.strptime(timestamp, format)
            
            # Try common formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
        
        return datetime.utcnow()
    
    def parse_log_entry(self, log_event: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single log entry.
        
        Args:
            log_event: Raw log event from CloudWatch
            
        Returns:
            Parsed and enriched log entry
        """
        message = log_event.get('message', '')
        timestamp = log_event.get('timestamp', datetime.utcnow())
        log_stream = log_event.get('log_stream', '')
        
        # Try parsing as JSON
        json_data = self.parse_json_log(message)
        
        if json_data:
            # Extract fields from JSON
            parsed = {
                'timestamp': self.parse_timestamp(timestamp),
                'message': message,
                'log_stream': log_stream,
                'service_name': self.extract_service_name(log_stream),
                'log_level': json_data.get('level', 'INFO'),
                'request_id': json_data.get('requestId'),
                'duration': json_data.get('duration'),
                'memory_used': json_data.get('memoryUsed'),
                'error_type': json_data.get('errorType'),
                'error_message': json_data.get('errorMessage'),
                'raw_json': json_data
            }
        else:
            # Plain text log
            parsed = {
                'timestamp': self.parse_timestamp(timestamp),
                'message': message,
                'log_stream': log_stream,
                'service_name': self.extract_service_name(log_stream),
                'log_level': self.extract_log_level(message),
                'request_id': None,
                'duration': None,
                'memory_used': None,
                'error_type': None,
                'error_message': None,
                'raw_json': None
            }
        
        return parsed
    
    def parse_logs_to_dataframe(
        self,
        log_events: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Parse multiple log events to DataFrame.
        
        Args:
            log_events: List of raw log events
            
        Returns:
            DataFrame with parsed logs
        """
        parsed_logs = [
            self.parse_log_entry(event) 
            for event in log_events
        ]
        
        df = pd.DataFrame(parsed_logs)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Parsed {len(df)} log entries into DataFrame")
        
        return df


if __name__ == "__main__":
    # Example usage
    parser = LogParser()
    
    # Example log event
    sample_event = {
        'timestamp': datetime.now(),
        'message': '{"level": "ERROR", "message": "Connection timeout", "duration": 5000}',
        'log_stream': '/aws/lambda/my-function'
    }
    
    parsed = parser.parse_log_entry(sample_event)
    logger.info(f"Parsed log: {parsed}")