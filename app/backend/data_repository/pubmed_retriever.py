from typing import List, Dict, Any, Optional
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
import time
from backend.data_repository.abstract_retriever import AbstractRetriever
from backend.data_repository.models import ScientificAbstract

class PubMedRetriever(AbstractRetriever):
    """Concrete implementation for retrieving abstracts from PubMed"""
    
    def __init__(self, max_results: int = 50, email: Optional[str] = None):
        super().__init__(max_results)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = email  # Required for NCBI API etiquette
        self.api_key = None  # Can be set for higher rate limits
        
    def get_abstract_data(self, scientist_question: str) -> List[ScientificAbstract]:
        """
        Retrieve abstracts from PubMed based on a research question
        """
        if not self.validate_query(scientist_question):
            logger.warning(f"Invalid query: {scientist_question}")
            return []
        
        try:
            # Step 1: Search for relevant PMIDs
            pmids = self._search_pubmed(scientist_question)
            
            if not pmids:
                logger.info(f"No results found for query: {scientist_question}")
                return []
            
            # Step 2: Fetch detailed information for PMIDs
            abstracts = self._fetch_abstracts(pmids)
            
            logger.info(f"Retrieved {len(abstracts)} abstracts for query: {scientist_question}")
            return abstracts
            
        except Exception as e:
            logger.error(f"Error retrieving abstracts: {str(e)}")
            return []
    
    def search_by_keywords(self, keywords: List[str]) -> List[ScientificAbstract]:
        """Search PubMed using specific keywords"""
        if not keywords:
            return []
        
        # Construct search query from keywords
        query = " AND ".join([f'"{keyword}"' for keyword in keywords])
        return self.get_abstract_data(query)
    
    def _search_pubmed(self, query: str) -> List[str]:
        """Search PubMed and return list of PMIDs"""
        search_url = f"{self.base_url}/esearch.fcgi"
        
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': self.max_results,
            'retmode': 'xml',
            'usehistory': 'y'
        }
        
        if self.email:
            params['email'] = self.email
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            pmids = [id_elem.text for id_elem in root.findall('.//Id')]
            
            return pmids[:self.max_results]
            
        except requests.RequestException as e:
            logger.error(f"PubMed search request failed: {str(e)}")
            return []
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed search response: {str(e)}")
            return []
    
    def _fetch_abstracts(self, pmids: List[str]) -> List[ScientificAbstract]:
        """Fetch detailed abstract information for given PMIDs"""
        if not pmids:
            return []
        
        fetch_url = f"{self.base_url}/efetch.fcgi"
        
        # PubMed allows fetching multiple articles at once
        pmid_string = ",".join(pmids)
        
        params = {
            'db': 'pubmed',
            'id': pmid_string,
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        
        if self.email:
            params['email'] = self.email
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = requests.get(fetch_url, params=params, timeout=60)
            response.raise_for_status()
            
            # Parse XML response
            abstracts = self._parse_pubmed_xml(response.content)
            
            # Add rate limiting to be respectful to NCBI
            time.sleep(0.1)
            
            return abstracts
            
        except requests.RequestException as e:
            logger.error(f"PubMed fetch request failed: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error parsing PubMed response: {str(e)}")
            return []
    
    def _parse_pubmed_xml(self, xml_content: bytes) -> List[ScientificAbstract]:
        """Parse PubMed XML response into ScientificAbstract objects"""
        abstracts = []
        
        try:
            root = ET.fromstring(xml_content)
            articles = root.findall('.//PubmedArticle')
            
            for article in articles:
                try:
                    abstract_data = self._extract_article_data(article)
                    if abstract_data:
                        abstract_obj = ScientificAbstract(**abstract_data)
                        abstracts.append(abstract_obj)
                except Exception as e:
                    logger.warning(f"Failed to parse individual article: {str(e)}")
                    continue
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed XML: {str(e)}")
        
        return abstracts
    
    def _extract_article_data(self, article_elem) -> Optional[Dict[str, Any]]:
        """Extract data from a single PubMed article element"""
        try:
            # Extract PMID
            pmid_elem = article_elem.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            # Extract title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else None
            
            # Extract abstract
            abstract_elem = article_elem.find('.//AbstractText')
            abstract_content = abstract_elem.text if abstract_elem is not None else None
            
            # Skip if no abstract content
            if not abstract_content:
                return None
            
            # Extract authors
            author_elems = article_elem.findall('.//Author')
            authors = []
            for author_elem in author_elems:
                lastname = author_elem.find('LastName')
                firstname = author_elem.find('ForeName')
                if lastname is not None:
                    name = lastname.text
                    if firstname is not None:
                        name = f"{firstname.text} {name}"
                    authors.append(name)
            
            # Extract publication year
            year_elem = article_elem.find('.//PubDate/Year')
            year = int(year_elem.text) if year_elem is not None else None
            
            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else None
            
            # Extract DOI
            doi_elems = article_elem.findall('.//ArticleId[@IdType="doi"]')
            doi = doi_elems[0].text if doi_elems else None
            
            # Extract keywords
            keyword_elems = article_elem.findall('.//Keyword')
            keywords = [kw.text for kw in keyword_elems if kw.text]
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract_content': abstract_content,
                'authors': authors,
                'year': year,
                'journal': journal,
                'doi': doi,
                'keywords': keywords,
                'source': 'PubMed',
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
            }
            
        except Exception as e:
            logger.warning(f"Error extracting article data: {str(e)}")
            return None
