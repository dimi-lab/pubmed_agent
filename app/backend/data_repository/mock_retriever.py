from typing import List
from backend.data_repository.abstract_retriever import AbstractRetriever
from backend.data_repository.models import ScientificAbstract

class MockAbstractRetriever(AbstractRetriever):
    """Mock implementation for testing and offline development"""
    
    def __init__(self, max_results: int = 50):
        super().__init__(max_results)
        self.mock_data = self._generate_mock_abstracts()
    
    def get_abstract_data(self, scientist_question: str) -> List[ScientificAbstract]:
        """Return mock abstracts based on query keywords"""
        if not self.validate_query(scientist_question):
            return []
        
        # Simple keyword matching for mock data
        query_lower = scientist_question.lower()
        relevant_abstracts = []
        
        for abstract in self.mock_data:
            # Check if query keywords appear in title or abstract
            if (any(keyword in abstract.title.lower() for keyword in query_lower.split()) or
                any(keyword in abstract.abstract_content.lower() for keyword in query_lower.split())):
                relevant_abstracts.append(abstract)
        
        return relevant_abstracts[:self.max_results]
    
    def search_by_keywords(self, keywords: List[str]) -> List[ScientificAbstract]:
        """Search mock data by keywords"""
        if not keywords:
            return []
        
        relevant_abstracts = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for abstract in self.mock_data:
            if any(kw in abstract.abstract_content.lower() or kw in abstract.title.lower() 
                   for kw in keywords_lower):
                relevant_abstracts.append(abstract)
        
        return relevant_abstracts[:self.max_results]
    
    def _generate_mock_abstracts(self) -> List[ScientificAbstract]:
        """Generate sample abstracts for testing"""
        return [
            ScientificAbstract(
                title="Advanced imaging techniques for early detection of Alzheimer's disease",
                authors=["Smith, J.", "Johnson, M.", "Brown, K."],
                year=2023,
                abstract_content="Early detection of Alzheimer's disease remains a critical challenge in neurology. This study examines the efficacy of advanced neuroimaging techniques, including PET scans and MRI, in identifying biomarkers associated with early-stage Alzheimer's disease. Our findings suggest that combining multiple imaging modalities significantly improves diagnostic accuracy.",
                doi="10.1016/j.neuroimage.2023.001",
                journal="NeuroImage",
                keywords=["Alzheimer's", "neuroimaging", "biomarkers", "early detection"],
                source="Mock"
            ),
            ScientificAbstract(
                title="Gut microbiota and its role in type 2 diabetes pathogenesis",
                authors=["Garcia, L.", "Wang, X.", "Thompson, R."],
                year=2023,
                abstract_content="The gut-brain axis plays a crucial role in metabolic disorders. This research investigates the relationship between gut microbiota composition and type 2 diabetes development. We analyzed fecal samples from 200 participants and found significant differences in bacterial diversity between diabetic and healthy individuals.",
                doi="10.1038/s41467-023-001",
                journal="Nature Communications",
                keywords=["gut microbiota", "diabetes", "gut-brain axis", "metabolism"],
                source="Mock"
            ),
            ScientificAbstract(
                title="Mechanisms of resistance in targeted cancer therapy",
                authors=["Lee, S.", "Patel, A.", "Williams, D."],
                year=2023,
                abstract_content="Resistance to targeted cancer therapies represents a major clinical challenge. This review examines the molecular mechanisms underlying therapeutic resistance, including genetic mutations, epigenetic changes, and tumor microenvironment factors. Understanding these mechanisms is crucial for developing next-generation treatment strategies.",
                doi="10.1038/s41571-023-001",
                journal="Nature Reviews Clinical Oncology",
                keywords=["cancer", "drug resistance", "targeted therapy", "oncology"],
                source="Mock"
            )
        ]