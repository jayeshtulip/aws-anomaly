"""Traffic router for A/B testing."""
import hashlib
import random
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import pickle
import numpy as np
from loguru import logger
from dataclasses import dataclass
from datetime import datetime
import redis
import json


@dataclass
class Variant:
    """Model variant configuration."""
    id: str
    name: str
    model_path: str
    scaler_path: Optional[str]
    weight: float
    params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Load model on initialization."""
        self.model = self._load_model()
        self.scaler = self._load_scaler() if self.scaler_path else None
    
    def _load_model(self):
        """Load model from disk."""
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_scaler(self):
        """Load scaler from disk."""
        if self.scaler_path and Path(self.scaler_path).exists():
            with open(self.scaler_path, 'rb') as f:
                return pickle.load(f)
        return None


class TrafficRouter:
    """Route traffic to different model variants for A/B testing."""
    
    def __init__(self, experiment_name: str, config_path: str = "ab_testing/experiment_config.yaml"):
        """Initialize router."""
        self.experiment_name = experiment_name
        self.config = self._load_config(config_path)
        self.variants = self._load_variants()
        self.redis_client = self._init_redis()
        
        logger.info(f"Initialized router for experiment: {experiment_name}")
        logger.info(f"Variants: {[v.name for v in self.variants]}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load experiment configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['experiments'][self.experiment_name]
    
    def _load_variants(self) -> List[Variant]:
        """Load all variants."""
        variants = []
        for v_config in self.config['variants']:
            variant = Variant(
                id=v_config['id'],
                name=v_config['name'],
                model_path=v_config.get('model_path'),
                scaler_path=v_config.get('scaler_path'),
                weight=v_config['weight'],
                params=v_config.get('params')
            )
            variants.append(variant)
        return variants
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis for tracking assignments."""
        try:
            client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            client.ping()
            return client
        except:
            logger.warning("Redis not available, using in-memory tracking")
            return None
    
    def assign_variant(self, user_id: str, use_sticky: bool = True) -> Variant:
        """Assign a variant to a user."""
        # Check for existing assignment (sticky sessions)
        if use_sticky and self.redis_client:
            assignment = self.redis_client.get(f"assignment:{self.experiment_name}:{user_id}")
            if assignment:
                variant_id = assignment
                return next(v for v in self.variants if v.id == variant_id)
        
        # Use consistent hashing for deterministic assignment
        hash_value = int(hashlib.md5(f"{user_id}{self.experiment_name}".encode()).hexdigest(), 16)
        random.seed(hash_value)
        
        # Weighted random selection
        variant = random.choices(
            self.variants,
            weights=[v.weight for v in self.variants],
            k=1
        )[0]
        
        # Store assignment
        if self.redis_client:
            self.redis_client.setex(
                f"assignment:{self.experiment_name}:{user_id}",
                86400 * 7,  # 7 days TTL
                variant.id
            )
        
        logger.debug(f"Assigned user {user_id} to variant {variant.name}")
        return variant
    
    def predict(self, user_id: str, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction using assigned variant."""
        start_time = datetime.now()
        
        # Assign variant
        variant = self.assign_variant(user_id)
        
        # Prepare features
        if variant.scaler:
            features_scaled = variant.scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # Make prediction
        prediction = variant.model.predict(features_scaled)[0]
        try:
            probabilities = variant.model.predict_proba(features_scaled)[0]
        except:
            probabilities = None
        
        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log metrics
        result = {
            'variant_id': variant.id,
            'variant_name': variant.name,
            'prediction': int(prediction),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'latency_ms': latency_ms,
            'timestamp': datetime.now().isoformat()
        }
        
        # Track in Redis
        if self.redis_client:
            self._track_prediction(user_id, result)
        
        return result
    
    def _track_prediction(self, user_id: str, result: Dict[str, Any]):
        """Track prediction metrics in Redis."""
        key = f"metrics:{self.experiment_name}:{result['variant_id']}:{datetime.now().strftime('%Y%m%d')}"
        
        # Increment counters
        self.redis_client.hincrby(key, 'total_predictions', 1)
        self.redis_client.hincrby(key, f"prediction_{result['prediction']}", 1)
        
        # Track latency
        self.redis_client.hincrbyfloat(key, 'total_latency', result['latency_ms'])
        
        # Set expiry
        self.redis_client.expire(key, 86400 * 30)  # 30 days
    
    def get_variant_stats(self, variant_id: str, days: int = 7) -> Dict[str, Any]:
        """Get statistics for a variant."""
        if not self.redis_client:
            return {}
        
        stats = {
            'total_predictions': 0,
            'avg_latency_ms': 0,
            'predictions_by_class': {}
        }
        
        for day_offset in range(days):
            date = (datetime.now() - timedelta(days=day_offset)).strftime('%Y%m%d')
            key = f"metrics:{self.experiment_name}:{variant_id}:{date}"
            
            day_stats = self.redis_client.hgetall(key)
            if day_stats:
                stats['total_predictions'] += int(day_stats.get('total_predictions', 0))
                stats['total_latency'] = stats.get('total_latency', 0) + float(day_stats.get('total_latency', 0))
                
                for k, v in day_stats.items():
                    if k.startswith('prediction_'):
                        class_label = k.split('_')[1]
                        stats['predictions_by_class'][class_label] = \
                            stats['predictions_by_class'].get(class_label, 0) + int(v)
        
        if stats['total_predictions'] > 0:
            stats['avg_latency_ms'] = stats['total_latency'] / stats['total_predictions']
        
        return stats


# Example usage
if __name__ == "__main__":
    from datetime import timedelta
    
    # Initialize router for model comparison
    router = TrafficRouter("model_comparison")
    
    # Simulate predictions
    for i in range(100):
        user_id = f"user_{i}"
        features = np.random.randn(37)  # Random features
        
        result = router.predict(user_id, features)
        print(f"User: {user_id}, Variant: {result['variant_name']}, "
              f"Prediction: {result['prediction']}, Latency: {result['latency_ms']:.2f}ms")
    
    # Get stats
    for variant in router.variants:
        stats = router.get_variant_stats(variant.id)
        print(f"\nStats for {variant.name}:")
        print(f"  Total predictions: {stats.get('total_predictions', 0)}")
        print(f"  Avg latency: {stats.get('avg_latency_ms', 0):.2f}ms")
        print(f"  Predictions: {stats.get('predictions_by_class', {})}")