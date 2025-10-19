"""Launch Optuna dashboard for monitoring optimization."""
import optuna
from optuna_dashboard import run_server
from loguru import logger
import argparse


def launch_dashboard(study_name: str, storage: str = "sqlite:///optuna.db", port: int = 8080):
    """Launch Optuna dashboard."""
    logger.info(f"Launching Optuna dashboard for study: {study_name}")
    logger.info(f"Storage: {storage}")
    logger.info(f"Dashboard URL: http://localhost:{port}")
    
    try:
        run_server(storage, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Optuna dashboard")
    parser.add_argument("--study", default="xgboost_anomaly_optimization", help="Study name")
    parser.add_argument("--storage", default="sqlite:///optuna.db", help="Storage URL")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    
    args = parser.parse_args()
    launch_dashboard(args.study, args.storage, args.port)