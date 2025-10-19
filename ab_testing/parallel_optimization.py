"""Run parallel hyperparameter optimization."""
import multiprocessing as mp
from hyperparameter_search import XGBoostOptimizer, DATA_PATH, OUTPUT_DIR
import pandas as pd
import optuna
from loguru import logger
from pathlib import Path
import json


def optimize_worker(metric: str, n_trials: int, queue: mp.Queue):
    """Worker function for parallel optimization."""
    logger.info(f"Starting optimization worker for metric: {metric}")
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df.drop(['is_anomaly', 'timestamp', 'message'], axis=1, errors='ignore')
    y = df['is_anomaly']
    
    # Initialize optimizer
    optimizer = XGBoostOptimizer(X, y, metric=metric)
    
    # Run optimization
    study_name = f"xgboost_{metric}_optimization"
    study = optimizer.optimize(n_trials=n_trials, study_name=study_name)
    
    # Save results
    result = {
        'metric': metric,
        'study_name': study_name,
        'best_score': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }
    
    queue.put(result)
    logger.success(f"✅ Worker completed for metric: {metric}")
    
    return result


def run_parallel_optimization(metrics=['f1', 'auc'], n_trials_per_metric=50):
    """Run optimization for multiple metrics in parallel."""
    logger.info(f"Starting parallel optimization for metrics: {metrics}")
    logger.info(f"Trials per metric: {n_trials_per_metric}")
    
    # Create queue for results
    queue = mp.Queue()
    
    # Create processes
    processes = []
    for metric in metrics:
        p = mp.Process(target=optimize_worker, args=(metric, n_trials_per_metric, queue))
        p.start()
        processes.append(p)
    
    # Wait for completion
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not queue.empty():
        results.append(queue.get())
    
    # Save summary
    summary = {
        'metrics_optimized': metrics,
        'n_trials_per_metric': n_trials_per_metric,
        'results': results
    }
    
    output_file = OUTPUT_DIR / "parallel_optimization_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.success(f"✅ Parallel optimization complete!")
    logger.info(f"Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = run_parallel_optimization(metrics=['f1', 'auc'], n_trials_per_metric=50)
    
    print("\n" + "="*60)
    print("PARALLEL OPTIMIZATION RESULTS")
    print("="*60)
    for result in results:
        print(f"\nMetric: {result['metric']}")
        print(f"  Best Score: {result['best_score']:.4f}")
        print(f"  Best Params: {result['best_params']}")