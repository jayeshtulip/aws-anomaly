"""
Embedding Pipeline for Log Data
Converts logs into vector embeddings for semantic search
"""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LogChunk:
    """Represents a chunk of log data with metadata"""
    text: str
    timestamp: str
    log_level: str
    source: str
    metadata: Dict


class EmbeddingPipeline:
    """Pipeline for embedding log data and storing in Qdrant"""
    
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "log_embeddings"
    ):
        """
        Initialize embedding pipeline
        
        Args:
            qdrant_url: URL of Qdrant service
            model_name: Name of sentence transformer model
            collection_name: Name of Qdrant collection
        """
        logger.info(f"Initializing embedding pipeline with model: {model_name}")
        
        # Initialize embedding model
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        
        # Create collection if it doesn't exist
        self._create_collection()
        
        logger.info("Embedding pipeline initialized successfully")
    
    def _create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def chunk_log_message(self, log_message: str, max_length: int = 512) -> List[str]:
        """
        Split long log messages into chunks
        
        Args:
            log_message: Log message to chunk
            max_length: Maximum length of each chunk
            
        Returns:
            List of log chunks
        """
        if len(log_message) <= max_length:
            return [log_message]
        
        # Split by sentences/lines
        chunks = []
        current_chunk = ""
        
        for line in log_message.split('\n'):
            if len(current_chunk) + len(line) <= max_length:
                current_chunk += line + '\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def embed_logs(self, log_chunks: List[LogChunk]) -> List[Dict]:
        """
        Create embeddings for log chunks
        
        Args:
            log_chunks: List of log chunks to embed
            
        Returns:
            List of embeddings with metadata
        """
        logger.info(f"Embedding {len(log_chunks)} log chunks")
        
        # Extract text from chunks
        texts = [chunk.text for chunk in log_chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Combine embeddings with metadata
        results = []
        for chunk, embedding in zip(log_chunks, embeddings):
            # Generate unique ID from content hash
            chunk_id = hashlib.md5(chunk.text.encode()).hexdigest()
            
            results.append({
                'id': chunk_id,
                'vector': embedding.tolist(),
                'payload': {
                    'text': chunk.text,
                    'timestamp': chunk.timestamp,
                    'log_level': chunk.log_level,
                    'source': chunk.source,
                    'metadata': chunk.metadata
                }
            })
        
        logger.info(f"Created {len(results)} embeddings")
        return results
    
    def store_embeddings(self, embeddings: List[Dict]):
        """
        Store embeddings in Qdrant
        
        Args:
            embeddings: List of embeddings with metadata
        """
        logger.info(f"Storing {len(embeddings)} embeddings in Qdrant")
        
        try:
            points = [
                PointStruct(
                    id=emb['id'],
                    vector=emb['vector'],
                    payload=emb['payload']
                )
                for emb in embeddings
            ]
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info("Embeddings stored successfully")
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise
    
    def search_similar_logs(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar log entries
        
        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of similar log entries with scores
        """
        logger.info(f"Searching for logs similar to: {query[:100]}...")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                'text': result.payload['text'],
                'score': result.score,
                'timestamp': result.payload['timestamp'],
                'log_level': result.payload['log_level'],
                'source': result.payload['source'],
                'metadata': result.payload['metadata']
            })
        
        logger.info(f"Found {len(results)} similar logs")
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}


def process_anomaly_logs(
    pipeline: EmbeddingPipeline,
    anomaly_logs: List[Dict]
) -> None:
    """
    Process and embed anomaly detection logs
    
    Args:
        pipeline: Embedding pipeline instance
        anomaly_logs: List of anomaly log dictionaries
    """
    logger.info(f"Processing {len(anomaly_logs)} anomaly logs")
    
    # Convert to LogChunk objects
    log_chunks = []
    for log in anomaly_logs:
        # Chunk long messages
        chunks = pipeline.chunk_log_message(log.get('message', ''))
        
        for chunk in chunks:
            log_chunks.append(LogChunk(
                text=chunk,
                timestamp=log.get('timestamp', ''),
                log_level=log.get('level', 'INFO'),
                source=log.get('source', 'unknown'),
                metadata={
                    'prediction': log.get('prediction'),
                    'confidence': log.get('confidence'),
                    'features': log.get('features', {})
                }
            ))
    
    # Generate embeddings
    embeddings = pipeline.embed_logs(log_chunks)
    
    # Store in Qdrant
    pipeline.store_embeddings(embeddings)
    
    # Print stats
    stats = pipeline.get_collection_stats()
    logger.info(f"Collection stats: {stats}")


if __name__ == "__main__":
    # Example usage
    pipeline = EmbeddingPipeline(
        qdrant_url="http://localhost:6333",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Example: Process some sample logs
    sample_logs = [
        {
            'message': 'Database connection timeout after 30 seconds',
            'timestamp': '2025-10-18T10:00:00Z',
            'level': 'ERROR',
            'source': 'database',
            'prediction': 1,
            'confidence': 0.95
        },
        {
            'message': 'High CPU usage detected: 95% utilization',
            'timestamp': '2025-10-18T10:01:00Z',
            'level': 'WARNING',
            'source': 'system',
            'prediction': 1,
            'confidence': 0.87
        }
    ]
    
    process_anomaly_logs(pipeline, sample_logs)
    
    # Search example
    results = pipeline.search_similar_logs(
        query="connection problems",
        limit=3
    )
    
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Level: {result['log_level']}")