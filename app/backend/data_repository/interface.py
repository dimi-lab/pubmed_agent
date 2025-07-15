from abc import ABC, abstractmethod
from typing import List, Dict
from langchain_core.documents.base import Document
from backend.data_repository.models import UserQueryRecord, ScientificAbstract

class UserQueryDataStore(ABC):
    """Repository interface for interaction with abstract database"""
    
    @abstractmethod
    def save_dataset(self, abstracts_data: List[ScientificAbstract], user_query: str) -> str:
        """
        Save abstracts and details about the query to data storage.
        
        Args:
            abstracts_data: List of scientific abstracts
            user_query: Original user query string
            
        Returns:
            str: Newly assigned query ID
        """
        raise NotImplementedError
    
    @abstractmethod
    def read_dataset(self, query_id: str) -> List[ScientificAbstract]:
        """
        Retrieve abstracts for a given user query from data storage.
        
        Args:
            query_id: Unique identifier for the query
            
        Returns:
            List[ScientificAbstract]: Retrieved abstracts
        """
        raise NotImplementedError
    
    @abstractmethod
    def delete_dataset(self, query_id: str) -> None:
        """
        Delete all data for a given query id from the database.
        
        Args:
            query_id: Unique identifier for the query to delete
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_list_of_queries(self) -> Dict[str, str]:
        """
        Retrieve a dict with query id : user_query for UI display and lookup.
        
        Returns:
            Dict[str, str]: Dictionary mapping query IDs to query text
        """
        raise NotImplementedError
    
    def create_document_list(self, abstracts_data: List[ScientificAbstract]) -> List[Document]:
        """
        Convert ScientificAbstract objects to LangChain Document objects
        
        Args:
            abstracts_data: List of ScientificAbstract objects
            
        Returns:
            List[Document]: LangChain Document objects for RAG processing
        """
        documents = []
        
        for entry in abstracts_data:
            # Handle authors properly - convert list to string for metadata
            authors_str = ", ".join(entry.authors) if entry.authors else "Unknown"
            
            # Create document with comprehensive metadata
            doc = Document(
                page_content=entry.abstract_content,
                metadata={
                    "source": entry.doi or entry.pmid or "Unknown",
                    "title": entry.title or "Unknown Title",
                    "authors": authors_str,
                    "year_of_publication": entry.year,
                    "journal": entry.journal,
                    "doi": entry.doi,
                    "pmid": entry.pmid,
                    "url": entry.url,
                    "keywords": ", ".join(entry.keywords) if entry.keywords else "",
                    "content_length": len(entry.abstract_content)
                }
            )
            documents.append(doc)
        
        return documents
    
    def read_documents(self, query_id: str) -> List[Document]:
        """
        Read the dataset and convert it to LangChain Document objects
        
        Args:
            query_id: Unique identifier for the query
            
        Returns:
            List[Document]: LangChain Document objects ready for RAG
        """
        query_record = self.read_dataset(query_id)
        return self.create_document_list(query_record)
