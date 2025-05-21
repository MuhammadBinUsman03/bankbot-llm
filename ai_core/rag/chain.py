"""
RAG (Retrieval-Augmented Generation) chain implementation.
"""

from typing import List, Dict, Any, Optional
import logging
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from database import QdrantDB
from models import EmbeddingModel
from llm import LLMClient

logger = logging.getLogger(__name__)

class RagChain:
    """Class for building and executing RAG chains."""
    
    # Default RAG prompt template
    DEFAULT_PROMPT_TEMPLATE = """You are a helpful banking assistant. Use the following retrieved information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Retrieved information:
{context}

User Question: {question}

Your answer should be helpful, concise, accurate, and friendly.
"""
    
    def __init__(
        self, 
        db: QdrantDB,
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        collection_name: str = "qa_collection",
        prompt_template: Optional[str] = None,
        top_k: int = 3
    ):
        """
        Initialize the RAG chain.
        
        Args:
            db (QdrantDB): Qdrant database client
            embedding_model (EmbeddingModel): Embedding model
            llm_client (LLMClient): LLM client
            collection_name (str): Name of the collection to query
            prompt_template (Optional[str]): Custom prompt template
            top_k (int): Number of documents to retrieve
        """
        self.db = db
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.collection_name = collection_name
        self.top_k = top_k
        
        # Set prompt template
        template = prompt_template if prompt_template else self.DEFAULT_PROMPT_TEMPLATE
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Build chain
        self._build_chain()
        
        logger.info(f"Initialized RAG chain with collection: {collection_name}")
    
    def _retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from Qdrant vector database.
        
        Args:
            query (str): User query
            
        Returns:
            List[Dict[str, Any]]: List of relevant context items
        """
        # Generate embedding for the query
        query_vector = self.embedding_model.get_embedding(query)
        
        # Search in Qdrant
        search_results = self.db.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.top_k
        )
        
        # Format the results
        contexts = []
        for hit in search_results:
            contexts.append({
                "question": hit.payload.get("question", ""),
                "answer": hit.payload.get("answer", ""),
                "score": hit.score
            })
        
        return contexts
    
    def _format_context(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Format context for the prompt.
        
        Args:
            contexts (List[Dict[str, Any]]): List of context items
            
        Returns:
            str: Formatted context
        """
        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            formatted_contexts.append(
                f"[Context {i}]\nQuestion: {ctx['question']}\nAnswer: {ctx['answer']}\nRelevance Score: {ctx['score']:.4f}"
            )
        return "\n\n".join(formatted_contexts)
    
    def _build_chain(self):
        """Build the RAG chain with LangChain."""
        
        # Define how to get context
        def get_context(query):
            contexts = self._retrieve_context(query)
            return self._format_context(contexts)
        
        # Build the chain
        rag_chain = (
            {"context": get_context, "question": RunnablePassthrough()}
            | self.prompt
        )
        
        # Add a custom formatting step to handle message format conversion
        def format_for_llm(langchain_messages):
            if isinstance(langchain_messages, list) and len(langchain_messages) > 0:
                # Extract the content from LangChain messages
                content = langchain_messages[0].content if hasattr(langchain_messages[0], 'content') else str(langchain_messages[0])
                return [{"role": "user", "content": content}]
            else:
                return [{"role": "user", "content": str(langchain_messages)}]
        
        # Complete the chain with the LLM call
        self.chain = (
            rag_chain
            | format_for_llm
            | self.llm_client.generate_answer
            | StrOutputParser()
        )
    
    def answer(self, query: str) -> str:
        """
        Answer a query using the RAG chain.
        
        Args:
            query (str): User query
            
        Returns:
            str: Generated answer
        """
        try:
            return self.chain.invoke(query)
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            return "I'm sorry, I couldn't generate an answer due to an error."
