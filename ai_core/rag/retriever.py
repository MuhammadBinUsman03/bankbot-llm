"""
Retrieval functionality for RAG pipeline.
"""

from typing import List, Dict, Any, Optional
import logging

from ai_core.database import QdrantDB
from ai_core.models import EmbeddingModel

logger = logging.getLogger(__name__)

class VectorRetriever:
    """Class for retrieving relevant context from vector database."""
    
    def __init__(
        self, 
        db: Optional[QdrantDB] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        collection_name: str = "qa_collection",
        db_url: str = "http://localhost:6333"
    ):
        """
        Initialize the vector retriever.
        
        Args:
            db (Optional[QdrantDB]): Qdrant database client
            embedding_model (Optional[EmbeddingModel]): Embedding model
            collection_name (str): Name of the collection
            db_url (str): URL of the Qdrant server
        """
        # Initialize database client if not provided
        self.db = db if db else QdrantDB(url=db_url)
        
        # Initialize embedding model if not provided
        self.model = embedding_model if embedding_model else EmbeddingModel()
        
        self.collection_name = collection_name
        logger.info(f"Vector retriever initialized with collection: {collection_name}")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector database.
        
        Args:
            query (str): The query text
            top_k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of retrieved contexts with relevance scores
        """
        # Generate embedding for the query
        query_vector = self.model.get_embedding(query)
        
        # Search in Qdrant
        search_results = self.db.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        # Format the results
        contexts = []
        for hit in search_results:
            contexts.append({
                "question": hit.payload.get("question", ""),
                "answer": hit.payload.get("answer", ""),
                "score": hit.score
            })
        
        logger.info(f"Retrieved {len(contexts)} contexts for query: {query[:50]}...")
        return contexts
        
    def format_context(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Format retrieved contexts into a string for the prompt.
        
        Args:
            contexts (List[Dict[str, Any]]): List of retrieved contexts
            
        Returns:
            str: Formatted context string
        """
        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            formatted_contexts.append(
                f"[Context {i}]\nQuestion: {ctx['question']}\nAnswer: {ctx['answer']}\nRelevance Score: {ctx['score']:.4f}"
            )
        return "\n\n".join(formatted_contexts)