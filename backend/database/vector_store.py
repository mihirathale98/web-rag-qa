from sqlalchemy import select, func
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .models import WebPage, TextChunk, WebPageData, TextChunkData
from .db_client import db_client
from ..config import Config

class VectorStore:
    """Vector database operations for storing and retrieving text chunks"""
    
    def __init__(self, embedding_dimension: int = None):
        self.embedding_dimension = embedding_dimension or Config.EMBEDDING_DIMENSION
    
    def add_webpage(self, webpage_data: WebPageData, session: Session = None) -> WebPage:
        """Add or update a webpage record"""
        if session:
            return self._add_webpage_with_session(webpage_data, session)
        
        with db_client.get_session() as db_session:
            return self._add_webpage_with_session(webpage_data, db_session)
    
    def _add_webpage_with_session(self, webpage_data: WebPageData, session: Session) -> WebPage:
        """Add or update a webpage record with an existing session"""
        # Check if the webpage already exists
        existing = session.query(WebPage).filter(WebPage.url == webpage_data.url).first()
        
        if existing:
            # Update existing webpage
            existing.title = webpage_data.title or existing.title
            existing.last_crawled = func.now()
            existing.source_url = webpage_data.source_url or existing.source_url
            session.commit()
            return existing
        
        # Create new webpage
        webpage = WebPage(
            url=webpage_data.url,
            title=webpage_data.title,
            source_url=webpage_data.source_url
        )
        session.add(webpage)
        session.commit()
        session.refresh(webpage)
        return webpage
    
    def add_chunk(self, chunk_data: TextChunkData, session: Session = None) -> TextChunk:
        """Add a text chunk with embedding"""
        if session:
            return self._add_chunk_with_session(chunk_data, session)
        
        with db_client.get_session() as db_session:
            return self._add_chunk_with_session(chunk_data, db_session)
    
    def _add_chunk_with_session(self, chunk_data: TextChunkData, session: Session) -> TextChunk:
        """Add a text chunk with an existing session"""
        # Check if this chunk already exists
        existing = session.query(TextChunk).filter(
            TextChunk.web_page_id == chunk_data.web_page_id,
            TextChunk.chunk_index == chunk_data.chunk_index
        ).first()
        
        embedding_array = np.array(chunk_data.embedding) if chunk_data.embedding else None
        
        if existing:
            # Update existing chunk
            existing.content = chunk_data.content
            if embedding_array is not None:
                existing.embedding = embedding_array
            existing.metadata = chunk_data.metadata or existing.metadata
            session.commit()
            return existing
        
        # Create new chunk
        chunk = TextChunk(
            web_page_id=chunk_data.web_page_id,
            content=chunk_data.content,
            embedding=embedding_array,
            chunk_index=chunk_data.chunk_index,
            metadata=chunk_data.metadata
        )
        session.add(chunk)
        session.commit()
        session.refresh(chunk)
        return chunk
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[TextChunk, float]]:
        """Search for similar chunks based on vector similarity"""
        with db_client.get_session() as session:
            # Start building the query
            query = session.query(
                TextChunk,
                func.l2_distance(TextChunk.embedding, Vector(query_embedding)).label('distance')
            )
            
            # Apply filters if provided
            if filter_dict:
                for k, v in filter_dict.items():
                    if k == 'web_page_id':
                        query = query.filter(TextChunk.web_page_id == v)
                    elif k == 'metadata':
                        for mk, mv in v.items():
                            query = query.filter(TextChunk.metadata[mk].astext == str(mv))
            
            # Complete the query with order and limit
            results = query.order_by('distance').limit(top_k).all()
            
            # Convert to list of tuples (chunk, distance)
            return [(chunk, float(distance)) for chunk, distance in results]
    
    def get_webpage_by_url(self, url: str) -> Optional[WebPage]:
        """Get a webpage by URL"""
        with db_client.get_session() as session:
            return session.query(WebPage).filter(WebPage.url == url).first()
    
    def get_chunks_by_webpage_id(self, webpage_id: int) -> List[TextChunk]:
        """Get all chunks for a webpage"""
        with db_client.get_session() as session:
            return session.query(TextChunk).filter(
                TextChunk.web_page_id == webpage_id
            ).order_by(TextChunk.chunk_index).all()
    
    def delete_webpage(self, url: str) -> bool:
        """Delete a webpage and all its chunks"""
        with db_client.get_session() as session:
            webpage = session.query(WebPage).filter(WebPage.url == url).first()
            if webpage:
                session.delete(webpage)  # This will cascade delete chunks
                session.commit()
                return True
            return False