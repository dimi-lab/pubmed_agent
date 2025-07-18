from abc import ABC, abstractmethod
from typing import List
from backend.data_repository.models import ScientificAbstract

class AbstractRetriever(ABC):
    """Abstract base class for retrieving scientific abstracts"""
    
    @abstractmethod
    def get_abstract_data(self, scientist_question: str) -> List[ScientificAbstract]:
        """
        Retrieve a list of scientific abstracts based on a given query.
        
        Args:
            scientist_question (str): The research question or query
            
        Returns:
            List[ScientificAbstract]: List of relevant abstracts
        """
        raise NotImplementedError("Subclasses must implement get_abstract_data method")

