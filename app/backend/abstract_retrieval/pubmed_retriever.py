# backend/abstract_retrieval/pubmed_retriever.py
from typing import List, Optional
from metapub import PubMedFetcher
from metapub.core import PubMedArticle
from backend.data_repository.models import ScientificAbstract
from backend.abstract_retrieval.interface import AbstractRetriever
from backend.abstract_retrieval.pubmed_query_simplification import simplify_pubmed_query
from config.logging_config import get_logger
import time

class PubMedAbstractRetriever(AbstractRetriever):
    """Concrete implementation for retrieving abstracts from PubMed using metapub"""
    
    def __init__(self, pubmed_fetch_object: Optional[PubMedFetcher] = None, max_results: int = 50):
        """
        Initialize PubMed retriever
        
        Args:
            pubmed_fetch_object: Optional PubMedFetcher instance
            max_results: Maximum number of results to return
        """
        self.pubmed_fetch_object = pubmed_fetch_object or PubMedFetcher()
        self.max_results = max_results
        self.logger = get_logger(__name__)
        
    def _simplify_pubmed_query(self, query: str, simplification_function: callable = simplify_pubmed_query) -> str:
        """
        Simplify the PubMed query using the provided function
        
        Args:
            query (str): Original query
            simplification_function: Function to simplify the query
            
        Returns:
            str: Simplified query
        """
        try:
            return simplification_function(query)
        except Exception as e:
            self.logger.error(f"Error simplifying query '{query}': {str(e)}")
            return query
    
    def _get_abstract_list(self, query: str, simplify_query: bool = True) -> List[str]:
        """
        Fetch a list of PubMed IDs for the given query.
        
        Args:
            query (str): Search query
            simplify_query (bool): Whether to simplify the query first
            
        Returns:
            List[str]: List of PubMed IDs
        """
        if simplify_query:
            self.logger.info(f'Trying to simplify scientist query: {query}')
            query_simplified = self._simplify_pubmed_query(query)
            if query_simplified != query:
                self.logger.info(f'Initial query simplified to: {query_simplified}')
                query = query_simplified
            else:
                self.logger.info('Initial query is simple enough and does not need simplification.')
        
        self.logger.info(f'Searching abstracts for query: {query}')
        
        try:
            # Get PMIDs for the query with a limit
            pmids = self.pubmed_fetch_object.pmids_for_query(query, retmax=self.max_results)
            self.logger.info(f'Found {len(pmids)} PMIDs for query: {query}')
            return pmids
            
        except Exception as e:
            self.logger.error(f'Error fetching PMIDs for query "{query}": {str(e)}')
            return []
    
    def _get_abstracts(self, pubmed_ids: List[str]) -> List[ScientificAbstract]:
        """
        Fetch PubMed abstracts for given PMIDs
        
        Args:
            pubmed_ids (List[str]): List of PubMed IDs
            
        Returns:
            List[ScientificAbstract]: List of scientific abstracts
        """
        if not pubmed_ids:
            self.logger.warning('No PubMed IDs provided for abstract fetching')
            return []
            
        self.logger.info(f'Fetching abstract data for {len(pubmed_ids)} pubmed_ids')
        scientific_abstracts = []
        
        for i, pmid in enumerate(pubmed_ids):
            try:
                self.logger.debug(f'Fetching abstract {i+1}/{len(pubmed_ids)}: PMID {pmid}')
                
                # Fetch article by PMID
                article = self.pubmed_fetch_object.article_by_pmid(pmid)
                
                # Skip if no abstract content
                if not article or not article.abstract:
                    self.logger.debug(f'No abstract found for PMID {pmid}')
                    continue
                
                # Extract and format abstract data
                abstract_formatted = self._format_abstract(article, pmid)
                if abstract_formatted:
                    scientific_abstracts.append(abstract_formatted)
                
                # Add small delay to be respectful to the API
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f'Error fetching abstract for PMID {pmid}: {str(e)}')
                continue
        
        self.logger.info(f'Successfully retrieved {len(scientific_abstracts)} abstracts.')
        return scientific_abstracts
    
    def _format_abstract(self, article: PubMedArticle, pmid: str) -> Optional[ScientificAbstract]:
        """
        Format a PubMed article into a ScientificAbstract object
        
        Args:
            article: PubMedArticle object
            pmid: PubMed ID
            
        Returns:
            ScientificAbstract or None if formatting fails
        """
        try:
            # Handle authors list properly
            authors = []
            if hasattr(article, 'authors') and article.authors:
                if isinstance(article.authors, list):
                    authors = [str(author).strip() for author in article.authors if author]
                elif isinstance(article.authors, str):
                    authors = [article.authors.strip()]
            
            # Create ScientificAbstract object
            abstract_formatted = ScientificAbstract(
                doi=getattr(article, 'doi', None),
                title=getattr(article, 'title', None),
                authors=authors,
                year=getattr(article, 'year', None),
                abstract_content=getattr(article, 'abstract', ''),
                pmid=pmid,
                journal=getattr(article, 'journal', None),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                source="PubMed"
            )
            
            return abstract_formatted
            
        except Exception as e:
            self.logger.error(f'Error formatting abstract for PMID {pmid}: {str(e)}')
            return None
    
    def get_abstract_data(self, scientist_question: str, simplify_query: bool = True) -> List[ScientificAbstract]:
        """
        Retrieve abstract list for scientist query.
        
        Args:
            scientist_question (str): The research question
            simplify_query (bool): Whether to simplify the query before searching
            
        Returns:
            List[ScientificAbstract]: List of relevant abstracts
        """
        try:
            # Get PMIDs for the query
            pmids = self._get_abstract_list(scientist_question, simplify_query)
            
            if not pmids:
                self.logger.warning(f'No PMIDs found for query: {scientist_question}')
                return []
            
            # Fetch abstracts for the PMIDs
            abstracts = self._get_abstracts(pmids)
            
            self.logger.info(f'Retrieved {len(abstracts)} abstracts for query: {scientist_question}')
            return abstracts
            
        except Exception as e:
            self.logger.error(f'Error in get_abstract_data for query "{scientist_question}": {str(e)}')
            return []
