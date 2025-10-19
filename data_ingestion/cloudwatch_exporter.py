"""Export CloudWatch logs to S3 for processing."""
import os
import time
from datetime import datetime, timedelta
from typing import Optional, List
import boto3
from botocore.exceptions import ClientError
from loguru import logger


class CloudWatchExporter:
    """Export CloudWatch logs to S3 bucket."""
    
    def __init__(
        self,
        log_group_name: str,
        s3_bucket: str,
        s3_prefix: str = "cloudwatch-exports",
        region: str = "us-east-1"
    ):
        """Initialize CloudWatch exporter.
        
        Args:
            log_group_name: CloudWatch log group name
            s3_bucket: S3 bucket for exports
            s3_prefix: S3 prefix for exported logs
            region: AWS region
        """
        self.log_group_name = log_group_name
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.region = region
        
        self.logs_client = boto3.client('logs', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        
        logger.info(f"Initialized CloudWatch exporter for {log_group_name}")
    
    def export_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        task_name: Optional[str] = None
    ) -> str:
        """Export logs from CloudWatch to S3.
        
        Args:
            start_time: Start time for log export
            end_time: End time for log export
            task_name: Custom name for export task
            
        Returns:
            Export task ID
        """
        # Default to last 24 hours
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        # Convert to milliseconds since epoch
        from_time = int(start_time.timestamp() * 1000)
        to_time = int(end_time.timestamp() * 1000)
        
        # Generate task name
        if task_name is None:
            task_name = f"export_{int(time.time())}"
        
        # S3 destination
        destination = f"{self.s3_prefix}/{task_name}"
        
        try:
            logger.info(
                f"Exporting logs from {start_time} to {end_time} "
                f"to s3://{self.s3_bucket}/{destination}"
            )
            
            response = self.logs_client.create_export_task(
                logGroupName=self.log_group_name,
                fromTime=from_time,
                to=to_time,
                destination=self.s3_bucket,
                destinationPrefix=destination
            )
            
            task_id = response['taskId']
            logger.info(f"Export task created: {task_id}")
            
            return task_id
            
        except ClientError as e:
            logger.error(f"Failed to create export task: {e}")
            raise
    
    def wait_for_export(self, task_id: str, timeout: int = 600) -> bool:
        """Wait for export task to complete.
        
        Args:
            task_id: Export task ID
            timeout: Maximum wait time in seconds
            
        Returns:
            True if export completed successfully
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.logs_client.describe_export_tasks(
                    taskId=task_id
                )
                
                if not response['exportTasks']:
                    logger.error(f"Export task {task_id} not found")
                    return False
                
                task = response['exportTasks'][0]
                status = task['status']['code']
                
                logger.info(f"Export task {task_id} status: {status}")
                
                if status == 'COMPLETED':
                    logger.success(f"Export task {task_id} completed")
                    return True
                elif status == 'FAILED':
                    logger.error(f"Export task {task_id} failed")
                    return False
                elif status in ['CANCELLED', 'CANCELLED_FAILED']:
                    logger.warning(f"Export task {task_id} was cancelled")
                    return False
                
                # Wait before checking again
                time.sleep(10)
                
            except ClientError as e:
                logger.error(f"Error checking export status: {e}")
                return False
        
        logger.error(f"Export task {task_id} timed out after {timeout}s")
        return False
    
    def list_recent_exports(self, limit: int = 10) -> List[dict]:
        """List recent export tasks.
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of export task details
        """
        try:
            response = self.logs_client.describe_export_tasks(
                limit=limit
            )
            return response.get('exportTasks', [])
        except ClientError as e:
            logger.error(f"Failed to list export tasks: {e}")
            return []
    
    def export_and_wait(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        timeout: int = 600
    ) -> bool:
        """Export logs and wait for completion.
        
        Args:
            start_time: Start time for log export
            end_time: End time for log export
            timeout: Maximum wait time in seconds
            
        Returns:
            True if export completed successfully
        """
        task_id = self.export_logs(start_time, end_time)
        return self.wait_for_export(task_id, timeout)


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    exporter = CloudWatchExporter(
        log_group_name="/aws/lambda/your-function-name",  # Change this
        s3_bucket=os.getenv("S3_BUCKET_NAME", "triton-models-71544"),
        s3_prefix="cloudwatch-exports"
    )
    
    # Export last 24 hours
    success = exporter.export_and_wait()
    
    if success:
        logger.success("Log export completed successfully")
    else:
        logger.error("Log export failed")