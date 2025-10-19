"""
RAG (Retrieval-Augmented Generation) Chain for Anomaly Explanation
Combines vector search with LLM for intelligent anomaly analysis
"""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import requests
from llm.embedding_pipeline import EmbeddingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyContext:
    """Context for anomaly explanation"""
    anomaly_log: str
    similar_logs: List[Dict]
    anomaly_score: float
    features: Dict


class RAGChain:
    """RAG chain for anomaly explanation using vector search + LLM"""
    
    def __init__(
        self,
        vllm_url: str = "http://localhost:8000",
        qdrant_url: str = "http://localhost:6333",
        model_name: str = "/models/llama2-7b-gptq"
    ):
        """
        Initialize RAG chain
        
        Args:
            vllm_url: URL of vLLM service
            qdrant_url: URL of Qdrant service
            model_name: Name of LLM model
        """
        self.vllm_url = vllm_url
        self.model_name = model_name
        
        # Initialize embedding pipeline
        self.embedding_pipeline = EmbeddingPipeline(
            qdrant_url=qdrant_url
        )
        
        logger.info("RAG chain initialized successfully")
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve similar logs from vector database
        
        Args:
            query: Query text (anomaly log)
            top_k: Number of similar logs to retrieve
            
        Returns:
            List of similar logs with context
        """
        logger.info(f"Retrieving context for: {query[:100]}...")
        
        similar_logs = self.embedding_pipeline.search_similar_logs(
            query=query,
            limit=top_k,
            score_threshold=0.6
        )
        
        return similar_logs
    
    def build_prompt(
        self,
        anomaly_context: AnomalyContext
    ) -> str:
        """
        Build prompt for LLM with retrieved context
        
        Args:
            anomaly_context: Context about the anomaly
            
        Returns:
            Formatted prompt string
        """
        # Format similar logs
        context_str = ""
        if anomaly_context.similar_logs:
            context_str = "\n\nSimilar past incidents:\n"
            for i, log in enumerate(anomaly_context.similar_logs, 1):
                context_str += f"{i}. [{log['log_level']}] {log['text'][:200]}\n"
                context_str += f"   Similarity: {log['score']:.2f}\n"
        
        # Build complete prompt
        prompt = f"""You are an expert system administrator analyzing log anomalies.

Current Anomaly:
Log: {anomaly_context.anomaly_log}
Anomaly Score: {anomaly_context.anomaly_score:.2f}
{context_str}

Based on the anomaly and similar past incidents, provide:
1. Root Cause Analysis: What likely caused this anomaly?
2. Impact Assessment: How severe is this issue?
3. Recommended Actions: What should be done to resolve it?
4. Prevention: How to prevent this in the future?

Be specific and actionable in your response.

Analysis:"""
        
        return prompt
    
    def generate_explanation(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generate explanation using LLM
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated explanation
        """
        logger.info("Generating explanation with LLM...")
        
        try:
            response = requests.post(
                f"{self.vllm_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": ["\n\n\n", "---"]
                },
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            explanation = result['choices'][0]['text'].strip()
            logger.info(f"Generated explanation ({len(explanation)} chars)")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Error: Unable to generate explanation - {str(e)}"
    
    def explain_anomaly(
        self,
        anomaly_log: str,
        anomaly_score: float,
        features: Optional[Dict] = None
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve context + generate explanation
        
        Args:
            anomaly_log: The anomalous log message
            anomaly_score: Anomaly confidence score
            features: Optional feature data
            
        Returns:
            Dictionary with explanation and context
        """
        logger.info(f"Explaining anomaly: {anomaly_log[:100]}...")
        
        # Step 1: Retrieve similar logs
        similar_logs = self.retrieve_context(anomaly_log, top_k=3)
        
        # Step 2: Build context
        context = AnomalyContext(
            anomaly_log=anomaly_log,
            similar_logs=similar_logs,
            anomaly_score=anomaly_score,
            features=features or {}
        )
        
        # Step 3: Build prompt
        prompt = self.build_prompt(context)
        
        # Step 4: Generate explanation
        explanation = self.generate_explanation(prompt)
        
        return {
            'anomaly_log': anomaly_log,
            'anomaly_score': anomaly_score,
            'similar_incidents': len(similar_logs),
            'explanation': explanation,
            'retrieved_context': similar_logs,
            'prompt_used': prompt
        }


def demo_rag_pipeline():
    """Demonstrate RAG pipeline with example anomalies"""
    
    # Initialize RAG chain
    rag = RAGChain(
        vllm_url="http://localhost:8000",
        qdrant_url="http://localhost:6333"
    )
    
    # Example anomalies
    test_anomalies = [
        {
            'log': 'Database connection pool exhausted - 500 connections in use, max 500',
            'score': 0.95,
            'features': {'connections': 500, 'max_connections': 500}
        },
        {
            'log': 'Memory usage critical: 95% RAM utilized, swap at 80%',
            'score': 0.89,
            'features': {'ram_percent': 95, 'swap_percent': 80}
        },
        {
            'log': 'API response time degraded: p95 latency 5000ms (baseline: 200ms)',
            'score': 0.92,
            'features': {'p95_latency': 5000, 'baseline': 200}
        }
    ]
    
    # Process each anomaly
    results = []
    for anomaly in test_anomalies:
        print(f"\n{'='*80}")
        print(f"ANALYZING ANOMALY:")
        print(f"Log: {anomaly['log']}")
        print(f"Score: {anomaly['score']}")
        print(f"{'='*80}\n")
        
        result = rag.explain_anomaly(
            anomaly_log=anomaly['log'],
            anomaly_score=anomaly['score'],
            features=anomaly['features']
        )
        
        print(f"EXPLANATION:")
        print(result['explanation'])
        print(f"\nSimilar Incidents Found: {result['similar_incidents']}")
        
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Run demo
    results = demo_rag_pipeline()
    
    print(f"\n{'='*80}")
    print(f"Processed {len(results)} anomalies successfully!")
    print(f"{'='*80}")