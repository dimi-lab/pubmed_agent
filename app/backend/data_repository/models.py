from typing import Optional, List
from pydantic import BaseModel, Field, validator
from datetime import datetime

class ScientificAbstract(BaseModel):
    """Model for representing scientific abstracts with validation"""
    
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    title: Optional[str] = Field(None, min_length=1, description="Title of the paper")
    authors: Optional[List[str]] = Field(default_factory=list, description="List of author names")
    year: Optional[int] = Field(None, ge=1800, le=datetime.now().year + 1, description="Publication year")
    abstract_content: str = Field(..., min_length=10, description="Full abstract text content")
    journal: Optional[str] = Field(None, description="Journal name")
    keywords: Optional[List[str]] = Field(default_factory=list, description="Associated keywords")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    url: Optional[str] = Field(None, description="URL to full paper")
    source: Optional[str] = Field(default="Unknown", description="Database source")
    
    @validator('authors', pre=True)
    def validate_authors(cls, v):
        """Ensure authors is always a list"""
        if v is None:
            return []
        if isinstance(v, str):
            # Handle comma-separated author string
            return [author.strip() for author in v.split(',')]
        if isinstance(v, list):
            return [str(author).strip() for author in v if author]
        return []
    
    @validator('keywords', pre=True)
    def validate_keywords(cls, v):
        """Ensure keywords is always a list"""
        if v is None:
            return []
        if isinstance(v, str):
            return [kw.strip() for kw in v.split(',')]
        if isinstance(v, list):
            return [str(kw).strip() for kw in v if kw]
        return []
    
    @validator('doi')
    def validate_doi(cls, v):
        """Basic DOI format validation"""
        if v and not v.startswith(('10.', 'doi:', 'DOI:')):
            # Try to clean up DOI
            if '10.' in v:
                v = v[v.find('10.'):]
        return v
    
    def __str__(self):
        authors_str = ", ".join(self.authors[:3]) + ("..." if len(self.authors) > 3 else "")
        return f"{self.title} ({self.year}) - {authors_str}"
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        str_strip_whitespace = True

class UserQueryRecord(BaseModel):
    """Model for storing user query information"""
    
    user_query_id: str = Field(..., description="Unique identifier for the query")
    user_query: str = Field(..., min_length=1, description="Original user query text")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Timestamp when query was created")
    abstract_count: Optional[int] = Field(None, ge=0, description="Number of abstracts retrieved")
    
    class Config:
        validate_assignment = True
        str_strip_whitespace = True