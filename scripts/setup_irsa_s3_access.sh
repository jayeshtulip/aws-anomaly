#!/bin/bash

# Setup IRSA (IAM Roles for Service Accounts) for S3 Access
# This allows Kubernetes pods to access S3 without hardcoded credentials

set -e

CLUSTER_NAME="triton-inference-cluster"
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="755289151343"
S3_BUCKET="triton-models-71544"

echo "=== Setting up IRSA for S3 Access ==="

# Function to create service account with S3 access
create_service_account_with_s3() {
    local NAMESPACE=$1
    local SERVICE_ACCOUNT=$2
    local POLICY_NAME="${SERVICE_ACCOUNT}-s3-policy"
    
    echo "Creating service account: ${SERVICE_ACCOUNT} in namespace: ${NAMESPACE}"
    
    # Create IAM policy for S3 access
    cat > /tmp/${POLICY_NAME}.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${S3_BUCKET}",
                "arn:aws:s3:::${S3_BUCKET}/*"
            ]
        }
    ]
}
EOF

    # Create IAM policy
    aws iam create-policy \
        --policy-name ${POLICY_NAME} \
        --policy-document file:///tmp/${POLICY_NAME}.json \
        || echo "Policy ${POLICY_NAME} already exists"
    
    # Create service account with IAM role
    eksctl create iamserviceaccount \
        --cluster=${CLUSTER_NAME} \
        --namespace=${NAMESPACE} \
        --name=${SERVICE_ACCOUNT} \
        --attach-policy-arn=arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${POLICY_NAME} \
        --override-existing-serviceaccounts \
        --region=${AWS_REGION} \
        --approve
    
    rm /tmp/${POLICY_NAME}.json
}

# Create service accounts for different namespaces
create_service_account_with_s3 "mlflow" "mlflow-sa"
create_service_account_with_s3 "triton" "triton-sa"
create_service_account_with_s3 "ray" "ray-sa"
create_service_account_with_s3 "llm" "llm-sa"

echo "=== IRSA setup complete ==="