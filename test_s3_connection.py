"""Test S3 connection and permissions."""
import boto3
from botocore.exceptions import ClientError


def test_s3_connection():
    """Test S3 bucket access."""
    bucket_name = "triton-models-71544"
    
    # Create S3 client
    s3_client = boto3.client('s3', region_name='us-east-1')
    
    try:
        # Test 1: List bucket
        print(f"✓ Testing bucket access: {bucket_name}")
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
        print(f"✓ Bucket accessible. Found {response.get('KeyCount', 0)} objects")
        
        # Test 2: Upload test file
        print("\n✓ Testing write permissions...")
        test_content = b"Test content from anomaly detection pipeline"
        s3_client.put_object(
            Bucket=bucket_name,
            Key='test/connection_test.txt',
            Body=test_content
        )
        print("✓ Write successful")
        
        # Test 3: Read test file
        print("\n✓ Testing read permissions...")
        obj = s3_client.get_object(Bucket=bucket_name, Key='test/connection_test.txt')
        content = obj['Body'].read()
        assert content == test_content
        print("✓ Read successful")
        
        # Test 4: Delete test file
        print("\n✓ Testing delete permissions...")
        s3_client.delete_object(Bucket=bucket_name, Key='test/connection_test.txt')
        print("✓ Delete successful")
        
        # Test 5: List folders
        print("\n✓ Listing S3 folders:")
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Delimiter='/'
        )
        
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                print(f"  - {prefix['Prefix']}")
        
        print("\n✅ All S3 tests passed!")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"\n❌ S3 Error: {error_code}")
        print(f"   Message: {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    test_s3_connection()