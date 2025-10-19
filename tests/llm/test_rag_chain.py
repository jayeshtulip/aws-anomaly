"""
Tests for RAG chain
"""
"""
Tests for RAG chain
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llm.rag_chain import RAGChain, AnomalyContext

@pytest.fixture
def rag_chain():
    """Create RAG chain instance"""
    return RAGChain(
        vllm_url="http://localhost:8000",
        qdrant_url="http://localhost:6333"
    )


class TestRAGChain:
    """Test RAG chain functionality"""
    
    def test_rag_initialization(self, rag_chain):
        """Test RAG chain initializes correctly"""
        assert rag_chain.vllm_url == "http://localhost:8000"
        assert rag_chain.model_name == "/models/llama2-7b-gptq"
        assert rag_chain.embedding_pipeline is not None
    
    def test_retrieve_context(self, rag_chain):
        """Test context retrieval"""
        results = rag_chain.retrieve_context(
            query="database connection error",
            top_k=3
        )
        
        assert isinstance(results, list)
        assert len(results) <= 3
    
    def test_build_prompt(self, rag_chain):
        """Test prompt building"""
        context = AnomalyContext(
            anomaly_log="Database timeout after 30s",
            similar_logs=[
                {
                    'text': 'Database connection failed',
                    'score': 0.85,
                    'log_level': 'ERROR'
                }
            ],
            anomaly_score=0.95,
            features={'timeout': 30}
        )
        
        prompt = rag_chain.build_prompt(context)
        
        assert "Database timeout after 30s" in prompt
        assert "Database connection failed" in prompt
        assert "0.95" in prompt
        assert "Root Cause Analysis" in prompt
        assert "Recommended Actions" in prompt
    
    def test_generate_explanation(self, rag_chain):
        """Test LLM explanation generation"""
        prompt = """Analyze this anomaly:
Log: Database connection timeout
Score: 0.95

Provide root cause analysis."""
        
        explanation = rag_chain.generate_explanation(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 50
        # Should contain analysis keywords
        assert any(word in explanation.lower() for word in 
                  ['cause', 'issue', 'problem', 'timeout', 'database'])
    
    def test_explain_anomaly_complete(self, rag_chain):
        """Test complete anomaly explanation pipeline"""
        result = rag_chain.explain_anomaly(
            anomaly_log="Memory usage at 98%, system becoming unresponsive",
            anomaly_score=0.92,
            features={'memory_percent': 98}
        )
        
        assert 'anomaly_log' in result
        assert 'anomaly_score' in result
        assert 'explanation' in result
        assert 'retrieved_context' in result
        
        assert result['anomaly_score'] == 0.92
        assert len(result['explanation']) > 100
        assert 'memory' in result['explanation'].lower()
    
    def test_response_time_sla(self, rag_chain):
        """Test explanation generation meets SLA"""
        import time
        
        latencies = []
        for _ in range(10):
            start = time.time()
            rag_chain.explain_anomaly(
                anomaly_log="Test anomaly log message",
                anomaly_score=0.8
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        mean = sum(latencies) / len(latencies)
        
        print(f"\nLatency stats:")
        print(f"  Mean: {mean:.0f}ms")
        print(f"  P95: {p95:.0f}ms")
        
        # Adjusted for T4 GPU + network latency
        assert p95 < 15000  # P95 under 15 seconds
        assert mean < 12000  # Mean under 12 seconds


class TestOutputValidation:
    """Test LLM output quality"""
    
    def test_output_format(self, rag_chain):
        """Test explanation has expected sections"""
        result = rag_chain.explain_anomaly(
            anomaly_log="API latency increased to 2000ms",
            anomaly_score=0.88
        )
        
        explanation = result['explanation'].lower()
        
        # Should contain analysis sections
        keywords = ['cause', 'impact', 'action', 'recommend']
        matches = sum(1 for keyword in keywords if keyword in explanation)
        
        assert matches >= 2  # At least 2 expected keywords
    
    def test_output_length(self, rag_chain):
        """Test explanation length is reasonable"""
        result = rag_chain.explain_anomaly(
            anomaly_log="Disk space at 95% capacity",
            anomaly_score=0.90
        )
        
        explanation = result['explanation']
        
        # Should be substantive but not too long
        assert 100 < len(explanation) < 2000
    
    def test_no_hallucination(self, rag_chain):
        """Test LLM stays grounded in provided context"""
        result = rag_chain.explain_anomaly(
            anomaly_log="CPU usage at 100%",
            anomaly_score=0.91
        )
        
        explanation = result['explanation'].lower()
        
        # Should reference the actual anomaly
        assert 'cpu' in explanation or 'processor' in explanation