from .retriever_interface import AbstractRetriever
from .retriever_factory import RetrieverFactory

# Only import PubMedRetriever if metapub is available
try:
    from .pubmed_retriever import PubMedRetriever
    __all__ = ['AbstractRetriever', 'RetrieverFactory', 'PubMedRetriever']
except ImportError:
    __all__ = ['AbstractRetriever', 'RetrieverFactory']

# Always include mock retriever
from .mock_retriever import MockRetriever
__all__.append('MockRetriever')
