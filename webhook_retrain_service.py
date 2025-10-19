"""
Webhook service to trigger retraining on alerts from Prometheus/Alertmanager
"""
from flask import Flask, request, jsonify
import requests
import logging
import os
from datetime import datetime
import json

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = os.getenv('GITHUB_REPO', 'jayeshtulip/aws-anomaly')
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', '3SNDtjoGvOL1fZd4BWEYml2JcRx60FgX')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "webhook-retrain-service",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/trigger', methods=['POST'])
def trigger_retrain():
    """
    Receive alert from Alertmanager and trigger GitHub Actions workflow
    
    Expected payload from Alertmanager:
    {
        "alerts": [
            {
                "labels": {
                    "alertname": "DataDriftDetected",
                    "severity": "warning"
                },
                "annotations": {
                    "summary": "Data drift detected",
                    "description": "PSI score exceeded threshold"
                }
            }
        ]
    }
    """
    
    # Verify webhook secret for security
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        logger.warning("Missing Bearer token in Authorization header")
        return jsonify({"error": "Unauthorized - Missing Bearer token"}), 401
    
    token = auth_header[7:]  # Remove "Bearer " prefix
    if token != WEBHOOK_SECRET:
        logger.warning(f"Invalid webhook secret received")
        return jsonify({"error": "Unauthorized - Invalid token"}), 401
    
    # Parse alert payload
    try:
        alert_data = request.json
        if not alert_data:
            return jsonify({"error": "Empty payload"}), 400
        
        logger.info(f"Received webhook payload: {json.dumps(alert_data, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error parsing JSON payload: {e}")
        return jsonify({"error": "Invalid JSON payload"}), 400
    
    # Extract alerts
    alerts = alert_data.get('alerts', [])
    if not alerts:
        logger.warning("No alerts found in payload")
        return jsonify({"error": "No alerts in payload"}), 400
    
    # Process first alert (can be extended to handle multiple)
    alert = alerts[0]
    alert_name = alert.get('labels', {}).get('alertname', 'Unknown')
    severity = alert.get('labels', {}).get('severity', 'Unknown')
    description = alert.get('annotations', {}).get('description', 'No description')
    
    logger.info(f"Processing alert: {alert_name} (severity: {severity})")
    logger.info(f"Description: {description}")
    
    # Trigger GitHub Actions workflow
    try:
        result = trigger_github_workflow(alert_name, severity, description)
        
        return jsonify({
            "status": "success",
            "message": f"Retraining triggered for alert: {alert_name}",
            "alert_name": alert_name,
            "severity": severity,
            "workflow_triggered": result['event_type'],
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error triggering GitHub workflow: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "alert_name": alert_name
        }), 500

def trigger_github_workflow(alert_name, severity, description):
    """
    Trigger GitHub Actions workflow via repository_dispatch event
    
    Documentation: https://docs.github.com/en/rest/repos/repos#create-a-repository-dispatch-event
    """
    
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN environment variable not set")
    
    # Map alert names to event types
    event_type_mapping = {
        'DataDriftDetected': 'data-drift-alert',
        'ModelAccuracyDegraded': 'performance-drift-alert',
        'HighErrorRate': 'error-rate-alert',
        'HighLatency': 'latency-alert'
    }
    
    event_type = event_type_mapping.get(alert_name, 'generic-alert')
    
    # GitHub API endpoint
    url = f"https://api.github.com/repos/{GITHUB_REPO}/dispatches"
    
    # Headers
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Payload
    payload = {
        "event_type": event_type,
        "client_payload": {
            "alert_name": alert_name,
            "severity": severity,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "triggered_by": "alertmanager-webhook"
        }
    }
    
    logger.info(f"Triggering GitHub workflow: {event_type}")
    logger.info(f"Repository: {GITHUB_REPO}")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
    
    # Make API request
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    
    # Check response
    if response.status_code == 204:
        logger.info(f"Successfully triggered {event_type} workflow")
        return {
            "event_type": event_type,
            "status": "dispatched"
        }
    else:
        error_msg = f"GitHub API error: {response.status_code} - {response.text}"
        logger.error(error_msg)
        raise Exception(error_msg)

@app.route('/test', methods=['POST'])
def test_trigger():
    """Test endpoint to manually trigger retraining"""
    
    # Verify webhook secret
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer ') or auth_header[7:] != WEBHOOK_SECRET:
        return jsonify({"error": "Unauthorized"}), 401
    
    test_alert = request.json or {}
    alert_name = test_alert.get('alert_name', 'DataDriftDetected')
    severity = test_alert.get('severity', 'warning')
    description = test_alert.get('description', 'Manual test trigger')
    
    try:
        result = trigger_github_workflow(alert_name, severity, description)
        return jsonify({
            "status": "success",
            "message": "Test workflow triggered",
            "result": result
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    # Validate configuration
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN environment variable is not set!")
        logger.error("Please set it before running the service")
        exit(1)
    
    logger.info(f"Starting Webhook Retrain Service")
    logger.info(f"GitHub Repository: {GITHUB_REPO}")
    logger.info(f"Webhook Secret: {'*' * len(WEBHOOK_SECRET)}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)