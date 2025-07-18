"""
Corrected embeddings.py with proper CUDA device configuration (no typos)
"""

import torch
import os
import warnings
from langchain_core.embeddings import Embeddings
from typing import List

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_device():
    """Get the best available device for embeddings"""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        return device
    else:
        print("⚠️ CUDA not available, using CPU")
        return 'cpu'

def get_default_embeddings() -> Embeddings:
    """Get default embeddings model for RAG workflow"""
    
    # Get the correct device (cuda or cpu)
    device = get_device()
    
    # Try HuggingFace embeddings first
    try:
        # Try the new langchain-huggingface package first
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            print("✅ Using langchain-huggingface package")
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print("⚠️ Using langchain-community package (deprecated)")
        
        print(f"🔄 Loading sentence-transformers embeddings on {device}...")
        
        # Create embeddings with correct device
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device': device,  # Make sure this is 'cuda' not 'cude'
                'trust_remote_code': False
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 64 if device == 'cuda' else 32
            }
        )
        
        # Test the embeddings
        test_embedding = embeddings.embed_query("test embedding")
        print(f"✅ HuggingFace embeddings loaded successfully on {device.upper()}!")
        print(f"✅ Embedding dimension: {len(test_embedding)}")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ HuggingFace embeddings failed: {e}")
        print("🔄 Trying fallback TFIDF embeddings...")
        
        # Fallback to TFIDF embeddings
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.pipeline import Pipeline
            import numpy as np
            
            class TfidfEmbeddings(Embeddings):
                def __init__(self, embedding_dim: int = 384):
                    self.embedding_dim = embedding_dim
                    self.is_fitted = False
                    
                    self.pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(
                            max_features=5000,
                            stop_words='english',
                            ngram_range=(1, 2),
                            min_df=1,
                            max_df=0.95,
                            sublinear_tf=True,
                            norm='l2'
                        )),
                        ('svd', TruncatedSVD(
                            n_components=embedding_dim,
                            random_state=42,
                            algorithm='randomized'
                        ))
                    ])
                    
                    self.corpus = []
                    print(f"✅ TFIDF embeddings initialized (dimension: {embedding_dim})")
                
                def fit(self, texts: List[str]):
                    self.corpus.extend(texts)
                    self.pipeline.fit(self.corpus)
                    self.is_fitted = True
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    if not self.is_fitted:
                        self.fit(texts)
                    embeddings_matrix = self.pipeline.transform(texts)
                    return embeddings_matrix.tolist()
                
                def embed_query(self, text: str) -> List[float]:
                    if not self.is_fitted:
                        self.fit([text])
                    embedding = self.pipeline.transform([text])
                    return embedding[0].tolist()
            
            embeddings = TfidfEmbeddings(embedding_dim=384)
            test_embedding = embeddings.embed_query("test embedding")
            print(f"✅ TFIDF fallback embeddings loaded successfully!")
            print(f"✅ Embedding dimension: {len(test_embedding)}")
            
            return embeddings
            
        except Exception as fallback_error:
            print(f"❌ TFIDF fallback also failed: {fallback_error}")
            raise RuntimeError(f"All embedding options failed! Last error: {fallback_error}")

def test_embeddings():
    """Test the embeddings functionality"""
    print("🧪 Testing embeddings functionality...")
    
    try:
        embeddings = get_default_embeddings()
        
        # Test single query
        print("📝 Testing single query...")
        test_embedding = embeddings.embed_query("This is a test query for biomedical research")
        print(f"✅ Single query test passed - dimension: {len(test_embedding)}")
        
        # Test document batch
        print("📚 Testing document batch...")
        test_docs = [
            "Cancer research focuses on understanding tumor biology",
            "Machine learning algorithms help analyze medical data", 
            "Gene therapy offers new treatment possibilities",
            "Immunotherapy enhances the body's natural defenses",
            "Biomarkers enable early disease detection"
        ]
        
        doc_embeddings = embeddings.embed_documents(test_docs)
        print(f"✅ Document batch test passed - {len(doc_embeddings)} embeddings generated")
        
        # Test similarity (basic check)
        print("🔍 Testing similarity calculation...")
        import numpy as np
        
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        query_embedding = embeddings.embed_query("cancer treatment research")
        similarities = []
        
        for i, doc_emb in enumerate(doc_embeddings):
            sim = cosine_similarity(query_embedding, doc_emb)
            similarities.append((sim, test_docs[i]))
        
        similarities.sort(reverse=True)
        print(f"✅ Most similar to 'cancer treatment research': {similarities[0][1][:50]}... (similarity: {similarities[0][0]:.3f})")
        
        print("\n🎉 All embedding tests passed!")
        return embeddings
        
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Create embeddings instance when imported
if __name__ == "__main__":
    # When run directly, test embeddings
    embeddings = test_embeddings()
else:
    # When imported, create embeddings instance
    try:
        embeddings = get_default_embeddings()
        print("✅ Embeddings ready for RAG workflow!")
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        embeddings = None