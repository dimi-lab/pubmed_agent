from typing import Optional
from backend.abstract_retrieval.retriever_interface import AbstractRetriever

class RetrieverFactory:
    """Factory for creating different types of abstract retrievers"""
    
    @staticmethod
    def create_retriever(retriever_type: str = "pubmed", 
                        max_results: int = 50,
                        **kwargs) -> AbstractRetriever:
        """
        Create an abstract retriever instance
        
        Args:
            retriever_type: Type of retriever ("pubmed", "mock")
            max_results: Maximum number of results to return
            **kwargs: Additional arguments for specific retrievers
        
        Returns:
            AbstractRetriever: Configured retriever instance
        """
        if retriever_type.lower() == "pubmed":
            try:
                from backend.abstract_retrieval.pubmed_retriever import PubMedRetriever
                return PubMedRetriever(max_results=max_results)
            except ImportError:
                raise ImportError("PubMed retriever requires 'metapub' package. Install with: pip install metapub")
        
        elif retriever_type.lower() == "mock":
            from backend.abstract_retrieval.mock_retriever import MockRetriever
            return MockRetriever(max_results=max_results)
        
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

