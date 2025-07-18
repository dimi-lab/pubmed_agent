from typing import List, Optional
import logging
import time
import os
import random
from functools import wraps
from backend.data_repository.models import ScientificAbstract
from backend.abstract_retrieval.retriever_interface import AbstractRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rate_limit_with_backoff(max_retries=3):
    """Decorator for rate limiting with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_retries):
                try:
                    # Rate limiting delay
                    if hasattr(self, '_last_request_time'):
                        elapsed = time.time() - self._last_request_time
                        min_interval = 1.0 / self.requests_per_second
                        if elapsed < min_interval:
                            time.sleep(min_interval - elapsed)
                    
                    result = func(self, *args, **kwargs)
                    self._last_request_time = time.time()
                    return result
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                        if attempt < max_retries - 1:
                            # Exponential backoff with jitter
                            delay = (2 ** attempt) * 5 + random.uniform(0, 5)
                            logger.warning(f"Rate limit hit, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                        else:
                            logger.error("Max retries exceeded due to rate limiting")
                            raise
                    else:
                        raise
            return None
        return wrapper
    return decorator

class PubMedRetriever(AbstractRetriever):
    """Enhanced PubMed retriever with rate limiting and API key support"""

    def __init__(self, max_results: int = 50, email: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize PubMed retriever with enhanced configuration

        Args:
            max_results: Maximum number of results to return
            email: Email for NCBI API etiquette (recommended)
            api_key: PubMed API key for higher rate limits
        """
        self.max_results = max_results
        self.email = email or os.getenv('PUBMED_EMAIL')
        self.api_key = api_key or os.getenv('PUBMED_API_KEY')
        self.fetcher = None
        self._last_request_time = 0
        
        # Configure rate limits based on API key availability
        if self.api_key:
            self.requests_per_second = 8  # Conservative: allows for 10/sec limit
            logger.info("✅ Using PubMed API key - higher rate limits enabled")
        else:
            self.requests_per_second = 2  # Conservative: allows for 3/sec limit
            logger.warning("⚠️ No API key found - using slower rate limits")
        
        # Validate email
        if not self.email:
            logger.warning("⚠️ No email provided - this may cause issues with NCBI API")
        
        self._initialize_fetcher()
        
        logger.info(f"📊 Rate limit configured: {self.requests_per_second} requests/second")

    def _initialize_fetcher(self):
        """Initialize PubMed fetcher with error handling and configuration"""
        try:
            from metapub import PubMedFetcher
            
            # Initialize with email if provided
            if self.email:
                self.fetcher = PubMedFetcher(email=self.email)
                logger.info(f"✅ PubMedFetcher initialized with email: {self.email}")
            else:
                self.fetcher = PubMedFetcher()
                logger.info("✅ PubMedFetcher initialized without email")
                
        except ImportError as e:
            logger.error(f"❌ metapub not installed: {e}")
            raise ImportError("Please install metapub: pip install metapub")
        except Exception as e:
            logger.error(f"❌ Failed to initialize PubMedFetcher: {e}")
            raise

    @rate_limit_with_backoff(max_retries=3)
    def _get_pmids_for_query(self, query: str) -> List[str]:
        """Get PMIDs with rate limiting"""
        try:
            pmids = self.fetcher.pmids_for_query(query, retmax=self.max_results)
            return pmids or []
        except Exception as e:
            logger.error(f"Error getting PMIDs for query '{query}': {str(e)}")
            raise

    def get_abstract_data(self, scientist_question: str) -> List[ScientificAbstract]:
        """
        Retrieve abstracts from PubMed with enhanced error handling and rate limiting

        Args:
            scientist_question: The research question or query

        Returns:
            List[ScientificAbstract]: List of relevant abstracts
        """
        if not self.fetcher:
            logger.error("❌ PubMedFetcher not initialized")
            return []

        if not scientist_question or not scientist_question.strip():
            logger.warning("⚠️ Empty query provided")
            return []

        try:
            logger.info(f"🔍 Searching for: {scientist_question}")

            # Get PMIDs for the query with rate limiting
            pmids = self._get_pmids_for_query(scientist_question)

            if not pmids:
                logger.info("ℹ️ No PMIDs found for the query")
                return []

            logger.info(f"📋 Found {len(pmids)} PMIDs")

            # Fetch abstracts for the PMIDs with enhanced processing
            abstracts = self._fetch_abstracts_from_pmids_enhanced(pmids)

            logger.info(f"✅ Successfully retrieved {len(abstracts)} abstracts")
            return abstracts

        except Exception as e:
            logger.error(f"💥 Error retrieving abstracts: {str(e)}")
            return []

    def _fetch_abstracts_from_pmids_enhanced(self, pmids: List[str]) -> List[ScientificAbstract]:
        """
        Enhanced method to fetch abstracts with batch processing and rate limiting

        Args:
            pmids: List of PubMed IDs

        Returns:
            List[ScientificAbstract]: List of abstracts
        """
        abstracts = []
        
        # Process in batches to be respectful to the API
        batch_size = 10 if self.api_key else 5
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(pmids) + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} PMIDs)")
            
            # Process each PMID in the batch
            for j, pmid in enumerate(batch):
                try:
                    logger.debug(f"📄 Fetching PMID {pmid} ({j+1}/{len(batch)} in batch)")
                    
                    # Apply rate limiting before each request
                    abstract = self._fetch_single_abstract_with_rate_limit(pmid)
                    
                    if abstract:
                        abstracts.append(abstract)
                        logger.debug(f"✅ Successfully processed PMID {pmid}")
                    else:
                        logger.debug(f"⏭️ No abstract content for PMID {pmid}")
                    
                    # Small delay between individual requests within batch
                    if j < len(batch) - 1:  # Don't delay after last item in batch
                        time.sleep(1.0 / self.requests_per_second)
                
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "rate limit" in error_str:
                        logger.warning(f"🚫 Rate limit hit on PMID {pmid}, implementing backoff")
                        # Wait longer and continue with next
                        time.sleep(30)
                        continue
                    else:
                        logger.warning(f"⚠️ Error processing PMID {pmid}: {str(e)}")
                        continue
            
            # Pause between batches
            if i + batch_size < len(pmids):
                inter_batch_delay = 2.0 if self.api_key else 5.0
                logger.debug(f"⏸️ Pausing {inter_batch_delay}s between batches")
                time.sleep(inter_batch_delay)
        
        return abstracts

    def _fetch_single_abstract_with_rate_limit(self, pmid: str) -> Optional[ScientificAbstract]:
        """Fetch a single abstract with rate limiting"""
        try:
            # Apply rate limiting
            if hasattr(self, '_last_request_time'):
                elapsed = time.time() - self._last_request_time
                min_interval = 1.0 / self.requests_per_second
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            
            # Fetch article by PMID
            article = self.fetcher.article_by_pmid(pmid)
            self._last_request_time = time.time()

            if not article:
                logger.debug(f"No article found for PMID {pmid}")
                return None

            # Check if abstract exists
            if not hasattr(article, 'abstract') or not article.abstract:
                logger.debug(f"No abstract for PMID {pmid}")
                return None

            # Extract article information
            abstract_data = self._extract_article_info(article, pmid)
            return abstract_data

        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str:
                raise  # Re-raise rate limit errors to be handled by caller
            else:
                logger.warning(f"Error fetching PMID {pmid}: {str(e)}")
                return None

    def _fetch_abstracts_from_pmids(self, pmids: List[str]) -> List[ScientificAbstract]:
        """
        Legacy method maintained for compatibility - delegates to enhanced version
        """
        return self._fetch_abstracts_from_pmids_enhanced(pmids)

    def _extract_article_info(self, article, pmid: str) -> Optional[ScientificAbstract]:
        """
        Extract information from a PubMed article with enhanced error handling

        Args:
            article: PubMed article object
            pmid: PubMed ID

        Returns:
            ScientificAbstract or None if extraction fails
        """
        try:
            # Extract basic information
            title = getattr(article, 'title', None)
            abstract_content = getattr(article, 'abstract', None)

            # Skip if no abstract content
            if not abstract_content:
                logger.debug(f"No abstract content for PMID {pmid}")
                return None

            # Extract authors with better handling
            authors = []
            if hasattr(article, 'authors') and article.authors:
                if isinstance(article.authors, list):
                    authors = [str(author).strip() for author in article.authors if author]
                elif isinstance(article.authors, str):
                    authors = [str(article.authors).strip()]

            # Extract other metadata with better error handling
            year = getattr(article, 'year', None)
            if year:
                try:
                    year = int(year)
                except (ValueError, TypeError):
                    logger.debug(f"Invalid year format for PMID {pmid}: {year}")
                    year = None

            doi = getattr(article, 'doi', None)
            journal = getattr(article, 'journal', None)

            # Extract keywords if available
            keywords = []
            if hasattr(article, 'keywords') and article.keywords:
                if isinstance(article.keywords, list):
                    keywords = [str(kw).strip() for kw in article.keywords if kw]
                elif isinstance(article.keywords, str):
                    keywords = [str(article.keywords).strip()]

            # Create ScientificAbstract object
            abstract = ScientificAbstract(
                doi=doi,
                title=title,
                authors=authors,
                year=year,
                abstract_content=abstract_content,
                pmid=pmid,
                journal=journal,
                keywords=keywords,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                source="PubMed"
            )

            return abstract

        except Exception as e:
            logger.error(f"❌ Error extracting article info for PMID {pmid}: {str(e)}")
            return None

    def search_by_keywords(self, keywords: List[str]) -> List[ScientificAbstract]:
        """
        Search abstracts by specific keywords
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List[ScientificAbstract]: List of relevant abstracts
        """
        if not keywords:
            logger.warning("No keywords provided for search")
            return []
        
        # Construct search query from keywords
        query = " AND ".join([f'"{keyword}"' for keyword in keywords])
        logger.info(f"🔍 Searching by keywords: {keywords}")
        logger.info(f"📝 Constructed query: {query}")
        
        return self.get_abstract_data(query)

# Configuration helper function
def create_configured_pubmed_retriever(max_results: int = 5) -> PubMedRetriever:
    """
    Create a properly configured PubMed retriever with environment variables
    
    Args:
        max_results: Maximum number of results to return
        
    Returns:
        PubMedRetriever: Configured retriever instance
    """
    email = os.getenv('PUBMED_EMAIL')
    api_key = os.getenv('PUBMED_API_KEY')
    
    if not email:
        logger.warning("⚠️ PUBMED_EMAIL not set in environment variables")
        logger.info("💡 Set PUBMED_EMAIL environment variable for better API compliance")
    
    if not api_key:
        logger.warning("⚠️ PUBMED_API_KEY not set in environment variables")
        logger.info("💡 Set PUBMED_API_KEY environment variable for higher rate limits")
    else:
        logger.info("🔑 API key found - enhanced rate limits will be used")
    
    return PubMedRetriever(
        max_results=max_results,
        email=email,
        api_key=api_key
    )