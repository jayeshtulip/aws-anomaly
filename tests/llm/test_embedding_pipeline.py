"""
Tests for embedding pipeline
"""
"""
Tests for embedding pipeline
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llm.embedding_pipeline import EmbeddingPipeline, LogChunk, process_anomaly_logs

@pytest.fixture
def embedding_pipeline():
    """Create embedding pipeline instance"""
    return EmbeddingPipeline(
        qdrant_url="http://localhost:6333",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


class TestEmbeddingPipeline:
    """Test embedding pipeline functionality"""
    
    def test_pipeline_initialization(self, embedding_pipeline):
        """Test pipeline initializes correctly"""
        assert embedding_pipeline.model is not None
        assert embedding_pipeline.qdrant_client is not None
        assert embedding_pipeline.collection_name == "log_embeddings"
        assert embedding_pipeline.embedding_dim == 384  # MiniLM dimension
    
    def test_chunk_log_message_short(self, embedding_pipeline):
        """Test chunking of short messages"""
        message = "Short log message"
        chunks = embedding_pipeline.chunk_log_message(message, max_length=100)
        
        assert len(chunks) == 1
        assert chunks[0] == message
    
    def test_chunk_log_message_long(self, embedding_pipeline):
        """Test chunking of long messages"""
        message = "Line 1\n" * 100  # Create long message
        chunks = embedding_pipeline.chunk_log_message(message, max_length=50)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 60  # Allow some buffer
    
    def test_embed_logs(self, embedding_pipeline):
        """Test embedding generation"""
        log_chunks = [
            LogChunk(
                text="Database connection failed",
                timestamp="2025-10-18T10:00:00Z",
                log_level="ERROR",
                source="database",
                metadata={"prediction": 1}
            ),
            LogChunk(
                text="High CPU usage detected",
                timestamp="2025-10-18T10:01:00Z",
                log_level="WARNING",
                source="system",
                metadata={"prediction": 1}
            )
        ]
        
        embeddings = embedding_pipeline.embed_logs(log_chunks)
        
        assert len(embeddings) == 2
        assert all('id' in emb for emb in embeddings)
        assert all('vector' in emb for emb in embeddings)
        assert all('payload' in emb for emb in embeddings)
        assert all(len(emb['vector']) == 384 for emb in embeddings)
    
    def test_store_and_search_embeddings(self, embedding_pipeline):
        """Test storing and searching embeddings"""
        # Store test data
        log_chunks = [
            LogChunk(
                text="Timeout connecting to database server",
                timestamp="2025-10-18T10:00:00Z",
                log_level="ERROR",
                source="database",
                metadata={"prediction": 1}
            )
        ]
        
        embeddings = embedding_pipeline.embed_logs(log_chunks)
        embedding_pipeline.store_embeddings(embeddings)
        
        # Search for similar
        results = embedding_pipeline.search_similar_logs(
            query="database connection problem",
            limit=5,
            score_threshold=0.5
        )
        
        assert len(results) > 0
        assert results[0]['score'] > 0.5
        assert 'database' in results[0]['text'].lower()
    
    def test_semantic_search_accuracy(self, embedding_pipeline):
        """Test semantic search finds relevant logs"""
        # Store diverse logs
        test_logs = [
            {
                'message': 'Database connection timeout after 30 seconds',
                'timestamp': '2025-10-18T10:00:00Z',
                'level': 'ERROR',
                'source': 'database',
                'prediction': 1,
                'confidence': 0.95
            },
            {
                'message': 'User login successful from 192.168.1.1',
                'timestamp': '2025-10-18T10:01:00Z',
                'level': 'INFO',
                'source': 'auth',
                'prediction': 0,
                'confidence': 0.1
            }
        ]
        
        process_anomaly_logs(embedding_pipeline, test_logs)
        
        # Search for database issues
        results = embedding_pipeline.search_similar_logs(
            query="database connection error",
            limit=2,
            score_threshold=0.6
        )
        
        # Should find database log, not login log
        assert len(results) > 0
        assert 'database' in results[0]['text'].lower()
        assert results[0]['log_level'] == 'ERROR'


class TestLogChunk:
    """Test LogChunk dataclass"""
    
    def test_log_chunk_creation(self):
        """Test creating log chunk"""
        chunk = LogChunk(
            text="Test message",
            timestamp="2025-10-18T10:00:00Z",
            log_level="INFO",
            source="test",
            metadata={"key": "value"}
        )
        
        assert chunk.text == "Test message"
        assert chunk.timestamp == "2025-10-18T10:00:00Z"
        assert chunk.log_level == "INFO"
        assert chunk.source == "test"
        assert chunk.metadata == {"key": "value"}