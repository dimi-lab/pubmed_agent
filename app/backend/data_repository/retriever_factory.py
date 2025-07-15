from typing import Optional
from backend.data_repository.abstract_retriever import AbstractRetriever
from backend.data_repository.pubmed_retriever import PubMedRetriever
from backend.data_repository.mock_retriever import MockAbstractRetriever

class RetrieverFactory:
    """Factory for creating different types of abstract retrievers"""
    
    @staticmethod
    def create_retriever(retriever_type: str = "pubmed", 
                        max_results: int = 50,
                        email: Optional[str] = None,
                        **kwargs) -> AbstractRetriever:
        """
        Create an abstract retriever instance
        
        Args:
            retriever_type: Type of retriever ("pubmed", "mock")
            max_results: Maximum number of results to return
            email: Email for API etiquette (PubMed)
            **kwargs: Additional arguments for specific retrievers
        
        Returns:
            AbstractRetriever: Configured retriever instance
        """
        if retriever_type.lower() == "pubmed":
            return PubMedRetriever(max_results=max_results, email=email)
        elif retriever_type.lower() == "mock":
            return MockAbstractRetriever(max_results=max_results)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")


# Usage example
if __name__ == "__main__":
    # Example usage
    from backend.data_repository.retriever_factory import RetrieverFactory
    
    # Create a PubMed retriever
    retriever = RetrieverFactory.create_retriever(
        retriever_type="pubmed",
        max_results=10,
        email="your.email@example.com"
    )
    
    # Or use mock for testing
    # retriever = RetrieverFactory.create_retriever("mock")
    
    # Search for abstracts
    abstracts = retriever.get_abstract_data("Alzheimer's disease biomarkers")
    
    for abstract in abstracts:
        print(f"Title: {abstract.title}")
        print(f"Authors: {', '.join(abstract.authors)}")
        print(f"Year: {abstract.year}")
        print(f"Abstract: {abstract.abstract_content[:200]}...")
        print("-" * 50)