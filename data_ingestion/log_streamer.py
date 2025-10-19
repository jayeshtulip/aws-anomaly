"""Stream CloudWatch logs in real-time."""
from datetime import datetime, timedelta
from typing import Optional, Generator, Dict, Any
import boto3
from loguru import logger


class LogStreamer:
    """Stream logs from CloudWatch in real-time."""
    
    def __init__(
        self,
        log_group_name: str,
        log_stream_name: Optional[str] = None,
        region: str = "us-east-1"
    ):
        """Initialize log streamer.
        
        Args:
            log_group_name: CloudWatch log group name
            log_stream_name: Specific log stream (optional)
            region: AWS region
        """
        self.log_group_name = log_group_name
        self.log_stream_name = log_stream_name
        self.region = region
        
        self.logs_client = boto3.client('logs', region_name=region)
        
        logger.info(f"Initialized log streamer for {log_group_name}")
    
    def list_log_streams(self, limit: int = 50) -> list:
        """List available log streams.
        
        Args:
            limit: Maximum number of streams to return
            
        Returns:
            List of log stream names
        """
        try:
            response = self.logs_client.describe_log_streams(
                logGroupName=self.log_group_name,
                orderBy='LastEventTime',
                descending=True,
                limit=limit
            )
            
            streams = [
                stream['logStreamName'] 
                for stream in response.get('logStreams', [])
            ]
            
            logger.info(f"Found {len(streams)} log streams")
            return streams
            
        except Exception as e:
            logger.error(f"Failed to list log streams: {e}")
            return []
    
    def get_log_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filter_pattern: Optional[str] = None,
        limit: int = 10000
    ) -> Generator[Dict[str, Any], None, None]:
        """Get log events from CloudWatch.
        
        Args:
            start_time: Start time for logs
            end_time: End time for logs
            filter_pattern: CloudWatch filter pattern
            limit: Maximum events to retrieve
            
        Yields:
            Log events as dictionaries
        """
        # Default to last hour
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(hours=1)
        
        start_millis = int(start_time.timestamp() * 1000)
        end_millis = int(end_time.timestamp() * 1000)
        
        kwargs = {
            'logGroupName': self.log_group_name,
            'startTime': start_millis,
            'endTime': end_millis,
            'limit': min(limit, 10000)  # CloudWatch max
        }
        
        if self.log_stream_name:
            kwargs['logStreamNames'] = [self.log_stream_name]
        
        if filter_pattern:
            kwargs['filterPattern'] = filter_pattern
        
        try:
            events_retrieved = 0
            next_token = None
            
            while events_retrieved < limit:
                if next_token:
                    kwargs['nextToken'] = next_token
                
                response = self.logs_client.filter_log_events(**kwargs)
                
                events = response.get('events', [])
                if not events:
                    break
                
                for event in events:
                    if events_retrieved >= limit:
                        break
                    
                    yield {
                        'timestamp': datetime.fromtimestamp(
                            event['timestamp'] / 1000
                        ),
                        'message': event['message'],
                        'log_stream': event.get('logStreamName'),
                        'ingestion_time': datetime.fromtimestamp(
                            event['ingestionTime'] / 1000
                        )
                    }
                    
                    events_retrieved += 1
                
                next_token = response.get('nextToken')
                if not next_token:
                    break
            
            logger.info(f"Retrieved {events_retrieved} log events")
            
        except Exception as e:
            logger.error(f"Failed to get log events: {e}")
    
    def tail_logs(
        self,
        follow: bool = True,
        filter_pattern: Optional[str] = None,
        refresh_interval: int = 5
    ) -> Generator[Dict[str, Any], None, None]:
        """Tail logs in real-time.
        
        Args:
            follow: Continue following logs
            filter_pattern: CloudWatch filter pattern
            refresh_interval: Seconds between checks
            
        Yields:
            Log events as they arrive
        """
        last_check = datetime.utcnow() - timedelta(minutes=5)
        
        while True:
            current_time = datetime.utcnow()
            
            # Get events since last check
            for event in self.get_log_events(
                start_time=last_check,
                end_time=current_time,
                filter_pattern=filter_pattern
            ):
                yield event
            
            if not follow:
                break
            
            last_check = current_time
            
            # Wait before next check
            import time
            time.sleep(refresh_interval)


if __name__ == "__main__":
    # Example usage
    streamer = LogStreamer(
        log_group_name="/aws/lambda/your-function-name"  # Change this
    )
    
    # List available streams
    streams = streamer.list_log_streams()
    logger.info(f"Available streams: {streams[:5]}")
    
    # Get recent events
    logger.info("Fetching recent log events...")
    for i, event in enumerate(streamer.get_log_events(limit=10)):
        logger.info(f"Event {i+1}: {event['message'][:100]}")