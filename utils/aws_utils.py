"""AWS utility functions."""
import os
import boto3
import pandas as pd
from io import StringIO, BytesIO
from typing import Optional
from loguru import logger
from botocore.exceptions import ClientError


class S3Handler:
    """Handle S3 operations."""
    
    def __init__(self, region: str = "us-east-1"):
        """Initialize S3 handler.
        
        Args:
            region: AWS region
        """
        self.s3_client = boto3.client('s3', region_name=region)
        self.region = region
    
    def parse_s3_path(self, s3_path: str) -> tuple:
        """Parse S3 path into bucket and key.
        
        Args:
            s3_path: S3 path (s3://bucket/key)
            
        Returns:
            Tuple of (bucket, key)
        """
        path = s3_path.replace('s3://', '')
        parts = path.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        return bucket, key
    
    def read_csv_from_s3(
        self,
        s3_path: str,
        **kwargs
    ) -> pd.DataFrame:
        """Read CSV from S3.
        
        Args:
            s3_path: S3 path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame
        """
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            logger.info(f"Reading CSV from s3://{bucket}/{key}")
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(BytesIO(obj['Body'].read()), **kwargs)
            logger.info(f"Loaded {len(df)} rows from S3")
            return df
        except ClientError as e:
            logger.error(f"Failed to read from S3: {e}")
            raise
    
    def write_csv_to_s3(
        self,
        df: pd.DataFrame,
        s3_path: str,
        **kwargs
    ):
        """Write DataFrame to S3 as CSV.
        
        Args:
            df: DataFrame to write
            s3_path: S3 destination path
            **kwargs: Additional arguments for df.to_csv
        """
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            logger.info(f"Writing CSV to s3://{bucket}/{key}")
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, **kwargs)
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=csv_buffer.getvalue()
            )
            logger.success(f"Uploaded {len(df)} rows to S3")
        except ClientError as e:
            logger.error(f"Failed to write to S3: {e}")
            raise
    
    def upload_file(self, local_path: str, s3_path: str):
        """Upload file to S3.
        
        Args:
            local_path: Local file path
            s3_path: S3 destination path
        """
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")
            self.s3_client.upload_file(local_path, bucket, key)
            logger.success("Uploaded to S3 successfully")
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise
    
    def download_file(self, s3_path: str, local_path: str):
        """Download file from S3.
        
        Args:
            s3_path: S3 source path
            local_path: Local destination path
        """
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.s3_client.download_file(bucket, key, local_path)
            logger.success("Downloaded from S3 successfully")
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            raise
    
    def list_objects(
        self,
        bucket: str,
        prefix: str = '',
        max_keys: int = 1000
    ) -> list:
        """List objects in S3 bucket.
        
        Args:
            bucket: S3 bucket name
            prefix: Prefix to filter objects
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = [
                obj['Key'] 
                for obj in response.get('Contents', [])
            ]
            
            logger.info(f"Found {len(objects)} objects in s3://{bucket}/{prefix}")
            return objects
        except ClientError as e:
            logger.error(f"Failed to list objects: {e}")
            return []
    
    def delete_object(self, s3_path: str):
        """Delete object from S3.
        
        Args:
            s3_path: S3 path to delete
        """
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            logger.info(f"Deleting s3://{bucket}/{key}")
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            logger.success("Deleted from S3 successfully")
        except ClientError as e:
            logger.error(f"Failed to delete from S3: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    s3_handler = S3Handler()
    
    # List objects
    bucket = os.getenv('S3_BUCKET_NAME', 'triton-models-71544')
    objects = s3_handler.list_objects(bucket, prefix='data/')
    
    for obj in objects[:5]:
        logger.info(f"  - {obj}")