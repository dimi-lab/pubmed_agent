import json
import os
import re
import shutil
from typing import Dict, List, Optional
from datetime import datetime
from backend.data_repository.models import UserQueryRecord, ScientificAbstract
from backend.data_repository.interface import UserQueryDataStore
from config.logging_config import get_logger

class LocalJSONStore(UserQueryDataStore):
    """Local JSON file storage implementation for testing and development"""

    def __init__(self, storage_folder_path: str):
        self.storage_folder_path = storage_folder_path
        self.index_file_path = os.path.join(storage_folder_path, 'index.json')
        self.logger = get_logger(__name__)
        self.metadata_index = None
        
        # Ensure storage directory exists
        os.makedirs(storage_folder_path, exist_ok=True)
        
        # Initialize or load index
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or load the metadata index"""
        try:
            if os.path.exists(self.index_file_path):
                with open(self.index_file_path, 'r', encoding='utf-8') as file:
                    self.metadata_index = json.load(file)
            else:
                self.metadata_index = {}
                self._save_index()
        except Exception as e:
            self.logger.error(f"Failed to initialize index: {e}")
            self.metadata_index = {}

    def _save_index(self):
        """Save the metadata index to file"""
        try:
            with open(self.index_file_path, 'w', encoding='utf-8') as file:
                json.dump(self.metadata_index, file, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")

    def get_new_query_id(self) -> str:
        """Compute a new query ID by incrementing previous query ID integer suffix by 1"""
        try:
            if not self.metadata_index:
                return 'query_1'
            
            keys = [k for k in self.metadata_index.keys() if k.startswith('query_')]
            if not keys:
                return 'query_1'
            
            # Extract numbers from query IDs
            numbers = []
            for k in keys:
                try:
                    number = int(k.split('_')[-1])
                    numbers.append(number)
                except ValueError:
                    continue
            
            if not numbers:
                return 'query_1'
            
            max_number = max(numbers)
            return f'query_{max_number + 1}'
            
        except Exception as e:
            self.logger.error(f"Error generating new query ID: {e}")
            # Fallback to timestamp-based ID
            timestamp = int(datetime.now().timestamp())
            return f'query_{timestamp}'

    def read_dataset(self, query_id: str) -> List[ScientificAbstract]:
        """Read dataset containing abstracts from local storage"""
        abstracts_file_path = os.path.join(self.storage_folder_path, query_id, 'abstracts.json')
        
        try:
            with open(abstracts_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            abstracts = []
            for abstract_record in data:
                try:
                    abstract = ScientificAbstract(**abstract_record)
                    abstracts.append(abstract)
                except Exception as e:
                    self.logger.warning(f"Failed to parse abstract record: {e}")
                    continue
            
            self.logger.info(f"Successfully read {len(abstracts)} abstracts for query {query_id}")
            return abstracts
            
        except FileNotFoundError:
            self.logger.error(f'The JSON file for query {query_id} was not found at {abstracts_file_path}')
            raise FileNotFoundError(f'Abstracts file not found for query {query_id}')
        except json.JSONDecodeError as e:
            self.logger.error(f'Invalid JSON in abstracts file for query {query_id}: {e}')
            raise ValueError(f'Invalid JSON format in abstracts file for query {query_id}')
        except Exception as e:
            self.logger.error(f'Unexpected error reading dataset for query {query_id}: {e}')
            raise RuntimeError(f'Failed to read dataset for query {query_id}: {e}')

    def save_dataset(self, abstracts_data: List[ScientificAbstract], user_query: str) -> str:
        """Save abstract dataset and query metadata to local storage"""
        if not abstracts_data:
            raise ValueError("Cannot save empty abstracts dataset")
        
        query_id = self.get_new_query_id()
        query_dir = os.path.join(self.storage_folder_path, query_id)
        
        try:
            # Create query directory
            os.makedirs(query_dir, exist_ok=True)
            
            # Prepare user query record
            user_query_details = UserQueryRecord(
                user_query_id=query_id,
                user_query=user_query,
                created_at=datetime.now(),
                abstract_count=len(abstracts_data)
            )

            # Save abstracts
            abstracts_file_path = os.path.join(query_dir, 'abstracts.json')
            with open(abstracts_file_path, "w", encoding='utf-8') as file:
                list_of_abstracts = [model.model_dump() for model in abstracts_data]
                json.dump(list_of_abstracts, file, indent=4, ensure_ascii=False)

            # Save query details
            query_details_file_path = os.path.join(query_dir, 'query_details.json')
            with open(query_details_file_path, "w", encoding='utf-8') as file:
                query_details_dict = user_query_details.model_dump()
                # Convert datetime to string for JSON serialization
                if 'created_at' in query_details_dict and isinstance(query_details_dict['created_at'], datetime):
                    query_details_dict['created_at'] = query_details_dict['created_at'].isoformat()
                json.dump(query_details_dict, file, indent=4, ensure_ascii=False)

            # Update index
            self.metadata_index[query_id] = user_query
            self._save_index()

            self.logger.info(f"Data for query ID {query_id} saved successfully with {len(abstracts_data)} abstracts")
            return query_id

        except Exception as e:
            self.logger.error(f"Failed to save dataset for query ID {query_id}: {e}")
            # Clean up partial save
            if os.path.exists(query_dir):
                shutil.rmtree(query_dir)
            raise RuntimeError(f"Failed to save dataset: {e}")
        
    def delete_dataset(self, query_id: str) -> None:
        """Delete abstracts dataset and query metadata from local storage"""
        path_to_data = os.path.join(self.storage_folder_path, query_id)
        
        try:
            if os.path.exists(path_to_data):
                shutil.rmtree(path_to_data)
                self.logger.info(f"Directory '{path_to_data}' has been deleted")
                
                # Remove from index
                if query_id in self.metadata_index:
                    del self.metadata_index[query_id]
                    self._save_index()
                    
            else:
                self.logger.warning(f"Directory '{path_to_data}' does not exist and cannot be deleted")
                
        except Exception as e:
            self.logger.error(f"Failed to delete dataset for query {query_id}: {e}")
            raise RuntimeError(f"Failed to delete dataset: {e}")

    def get_list_of_queries(self) -> Dict[str, str]:
        """Get a dictionary containing query ID (as key) and original user query (as value)"""
        if self.metadata_index is None:
            self._initialize_index()
        return self.metadata_index.copy()

    def _rebuild_index(self) -> Dict[str, str]:
        """Rebuild the index from all query details files"""
        index = {}
        
        try:
            if not os.path.exists(self.storage_folder_path):
                return index
            
            query_data_paths = [
                os.path.join(self.storage_folder_path, name) 
                for name in os.listdir(self.storage_folder_path)
                if os.path.isdir(os.path.join(self.storage_folder_path, name))
            ]
            
            for query_data_path in query_data_paths:
                metadata_path = os.path.join(query_data_path, 'query_details.json')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as file:
                            metadata = json.load(file)
                            query_id = metadata.get('user_query_id')
                            user_query = metadata.get('user_query')
                            
                            if query_id and user_query:
                                index[query_id] = user_query
                            else:
                                self.logger.warning(f"Invalid metadata in {metadata_path}")
                                
                    except Exception as e:
                        self.logger.warning(f"Failed to read metadata from {metadata_path}: {e}")
                else:
                    self.logger.warning(f"No query_details.json file found in {query_data_path}")
            
            # Save the rebuilt index
            self.metadata_index = index
            self._save_index()
            
            self.logger.info(f"Index rebuilt with {len(index)} entries")
            return index
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild index: {e}")
            return {}
