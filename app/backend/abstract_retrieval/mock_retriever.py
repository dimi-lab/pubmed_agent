from typing import List
from backend.abstract_retrieval.retriever_interface import AbstractRetriever
from backend.data_repository.models import ScientificAbstract

class MockRetriever(AbstractRetriever):
    """Mock retriever for testing and offline development"""
    
    def __init__(self, max_results: int = 50):
        self.max_results = max_results
        self.mock_data = self._create_mock_abstracts()
    
    def get_abstract_data(self, scientist_question: str) -> List[ScientificAbstract]:
        """Return mock abstracts based on query keywords"""
        if not scientist_question or not scientist_question.strip():
            return []
        
        # Simple keyword matching
        query_lower = scientist_question.lower()
        relevant_abstracts = []
        
        for abstract in self.mock_data:
            # Check if query keywords appear in title or abstract
            title_match = any(keyword in abstract.title.lower() for keyword in query_lower.split())
            content_match = any(keyword in abstract.abstract_content.lower() for keyword in query_lower.split())
            
            if title_match or content_match:
                relevant_abstracts.append(abstract)
        
        return relevant_abstracts[:self.max_results]
    
    def _create_mock_abstracts(self) -> List[ScientificAbstract]:
        """Generate sample abstracts for testing"""
        return [
            ScientificAbstract(
                title="Cancer Drug Resistance Mechanisms in Targeted Therapy",
                authors=["Smith, J.", "Johnson, M.", "Brown, K."],
                year=2023,
                abstract_content="Cancer drug resistance remains a major challenge in targeted therapy. This study examines molecular mechanisms underlying resistance development, including genetic mutations, epigenetic changes, and adaptive cellular responses. Our findings suggest that combination therapies targeting multiple pathways may overcome resistance mechanisms.",
                doi="10.1016/j.cancer.2023.001",
                journal="Cancer Research",
                pmid="12345678",
                source="Mock"
            ),
            ScientificAbstract(
                title="Gut Microbiota and Diabetes: Mechanisms and Therapeutic Targets",
                authors=["Garcia, L.", "Wang, X.", "Thompson, R."],
                year=2023,
                abstract_content="The gut microbiota plays a crucial role in diabetes pathogenesis through multiple mechanisms including metabolic regulation, immune modulation, and gut barrier function. This review examines current evidence for microbiota-based therapeutic interventions in diabetes management.",
                doi="10.1038/s41467-2023-002",
                journal="Nature Communications",
                pmid="23456789",
                source="Mock"
            ),
            ScientificAbstract(
                title="Alzheimer's Disease Biomarkers: From Discovery to Clinical Application",
                authors=["Lee, S.", "Patel, A.", "Williams, D."],
                year=2023,
                abstract_content="Early detection of Alzheimer's disease through biomarkers represents a critical advance in neurology. This study evaluates the clinical utility of amyloid-beta, tau proteins, and neuroimaging biomarkers for early diagnosis and disease monitoring in clinical practice.",
                doi="10.1038/s41571-2023-003",
                journal="Nature Reviews Neurology",
                pmid="34567890",
                source="Mock"
            )
        ]

