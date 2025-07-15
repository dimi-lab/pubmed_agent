from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from backend.data_repository.models import ScientificAbstract
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AbstractRetriever(ABC):
    """Abstract base class for retrieving scientific abstracts"""
    
    def __init__(self, max_results: int = 50):
        self.max_results = max_results
    
    @abstractmethod
    def get_abstract_data(self, scientist_question: str) -> List[ScientificAbstract]:
        """
        Retrieve a list of scientific abstracts based on a given query.
        
        Args:
            scientist_question (str): The research question or query
            
        Returns:
            List[ScientificAbstract]: List of relevant abstracts
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_abstract_data method")
    
    @abstractmethod
    def search_by_keywords(self, keywords: List[str]) -> List[ScientificAbstract]:
        """
        Search abstracts by specific keywords.
        
        Args:
            keywords (List[str]): List of keywords to search for
            
        Returns:
            List[ScientificAbstract]: List of relevant abstracts
        """
        raise NotImplementedError("Subclasses must implement search_by_keywords method")
    
    def filter_by_year(self, abstracts: List[ScientificAbstract], 
                      start_year: Optional[int] = None, 
                      end_year: Optional[int] = None) -> List[ScientificAbstract]:
        """Filter abstracts by publication year range"""
        filtered = []
        for abstract in abstracts:
            if abstract.year is None:
                continue
            if start_year and abstract.year < start_year:
                continue
            if end_year and abstract.year > end_year:
                continue
            filtered.append(abstract)
        return filtered
    
    def validate_query(self, query: str) -> bool:
        """Validate search query"""
        if not query or not isinstance(query, str):
            return False
        if len(query.strip()) < 3:
            return False
        return True