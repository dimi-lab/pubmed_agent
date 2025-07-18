from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import VectorStore

class RagWorkflow(ABC):
    """Interface for the RAG workflow"""
    
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
    
    @abstractmethod
    def create_vector_index_for_user_query(self, documents: List[Document], query_id: str) -> VectorStore:
        """
        Create vector index based on documents retrieved for a specific user query
        
        Args:
            documents: List of LangChain Document objects
            query_id: Unique identifier for the query
            
        Returns:
            VectorStore: Created vector store index
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_vector_index_by_user_query(self, query_id: str) -> VectorStore:
        """
        Get existing vector index from a query ID
        
        Args:
            query_id: Unique identifier for the query
            
        Returns:
            VectorStore: Retrieved vector store index
        """
        raise NotImplementedError
    
    @abstractmethod
    def delete_vector_index(self, query_id: str) -> bool:
        """
        Delete vector index for a specific query
        
        Args:
            query_id: Unique identifier for the query
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        raise NotImplementedError
