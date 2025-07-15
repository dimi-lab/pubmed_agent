from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
import os

def get_default_embeddings() -> Embeddings:
    """Get default embeddings model for RAG workflow"""
    try:
        # Use a lightweight model that works offline
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        print(f"Failed to load default embeddings: {e}")
        raise

# Create embeddings instance
embeddings = get_default_embeddings()

