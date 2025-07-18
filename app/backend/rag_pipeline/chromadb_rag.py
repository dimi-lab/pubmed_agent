from typing import List, Optional
import chromadb
from langchain_community.vectorstores import VectorStore, Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents.base import Document
from config.logging_config import get_logger
import os
from backend.rag_pipeline.rag_interface import RagWorkflow

class ChromaDbRag(RagWorkflow):
    """RAG workflow implementation using ChromaDB as vector store"""
    
    def __init__(self, persist_directory: str, embeddings: Embeddings):
        super().__init__(embeddings)
        self.persist_directory = persist_directory
        self.client = self._create_chromadb_client()
        self.logger = get_logger(__name__)
        
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
    
    def _create_chromadb_client(self):
        """Create ChromaDB persistent client"""
        try:
            return chromadb.PersistentClient(path=self.persist_directory)
        except Exception as e:
            self.logger.error(f"Failed to create ChromaDB client: {e}")
            raise
    
    def _sanitize_collection_name(self, query_id: str) -> str:
        """Sanitize collection name to meet ChromaDB requirements"""
        # ChromaDB collection names must be 3-63 characters, alphanumeric + hyphens/underscores
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', query_id)
        sanitized = sanitized[:63]  # Limit length
        
        if len(sanitized) < 3:
            sanitized = f"query_{sanitized}"
        
        return sanitized
    
    def create_vector_index_for_user_query(self, documents: List[Document], query_id: str) -> VectorStore:
        """Create Chroma vector index and set query ID as collection name"""
        if not documents:
            raise ValueError(f"Cannot create vector index for empty document list (query_id: {query_id})")
        
        collection_name = self._sanitize_collection_name(query_id)
        self.logger.info(f'Creating vector index for {query_id} (collection: {collection_name})')
        
        try:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(collection_name)
                self.logger.info(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass  # Collection doesn't exist, which is fine
            
            # Create new vector index
            index = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=self.client,
                collection_name=collection_name,
                persist_directory=self.persist_directory
            )
            
            self.logger.info(f'Successfully created vector index for {query_id} with {len(documents)} documents')
            return index
            
        except Exception as e:
            self.logger.error(f'Failed to create vector index for query: {query_id}. Error: {e}')
            raise RuntimeError(f"Vector index creation failed: {e}")
    
    def get_vector_index_by_user_query(self, query_id: str) -> VectorStore:
        """Retrieve existing Chroma index by collection name set to query ID"""
        collection_name = self._sanitize_collection_name(query_id)
        self.logger.info(f'Loading vector index for query: {query_id} (collection: {collection_name})')
        
        try:
            # Check if collection exists
            collections = self.client.list_collections()
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                raise ValueError(f"Collection {collection_name} does not exist for query {query_id}")
            
            # Load existing vector index
            index = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            self.logger.info(f'Successfully loaded vector index for {query_id}')
            return index
            
        except Exception as e:
            self.logger.error(f'Failed to retrieve vector index for query: {query_id}. Error: {e}')
            raise RuntimeError(f"Vector index retrieval failed: {e}")
    
    def delete_vector_index(self, query_id: str) -> bool:
        """Delete vector index for a specific query"""
        collection_name = self._sanitize_collection_name(query_id)
        self.logger.info(f'Deleting vector index for query: {query_id} (collection: {collection_name})')
        
        try:
            self.client.delete_collection(collection_name)
            self.logger.info(f'Successfully deleted vector index for {query_id}')
            return True
            
        except Exception as e:
            self.logger.warning(f'Failed to delete vector index for query: {query_id}. Error: {e}')
            return False
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return []