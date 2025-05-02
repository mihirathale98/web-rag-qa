from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

Base = declarative_base()

class WebPage(Base):
    """Model for storing web page metadata"""
    __tablename__ = "web_pages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(2048), nullable=False, unique=True, index=True)
    title = Column(String(512), nullable=True)
    last_crawled = Column(DateTime, default=datetime.utcnow)
    source_url = Column(String(2048), nullable=True)  # The URL that linked to this page
    
    # Relationships
    chunks = relationship("TextChunk", back_populates="web_page", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<WebPage(id={self.id}, url='{self.url}', title='{self.title}')>"


class TextChunk(Base):
    """Model for storing text chunks with their vector embeddings"""
    __tablename__ = "text_chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    web_page_id = Column(Integer, ForeignKey("web_pages.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # Using OpenAI's embedding dimension by default
    chunk_index = Column(Integer, nullable=False)  # Position of chunk in the document
    metadata = Column(JSONB, nullable=True)  # Additional metadata about the chunk
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    web_page = relationship("WebPage", back_populates="chunks")
    
    __table_args__ = (
        UniqueConstraint('web_page_id', 'chunk_index', name='uix_chunk_index'),
    )
    
    def __repr__(self):
        return f"<TextChunk(id={self.id}, web_page_id={self.web_page_id}, chunk_index={self.chunk_index})>"


class ChatMessage(Base):
    """Model for storing chat messages"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(64), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, conversation_id='{self.conversation_id}', role='{self.role}')>"


class ChatContext(Base):
    """Model for storing retrieval context for chat messages"""
    __tablename__ = "chat_contexts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False)
    chunk_ids = Column(ARRAY(Integer), nullable=False)  # IDs of the retrieved chunks
    relevance_scores = Column(ARRAY(Float), nullable=True)  # Relevance scores for each chunk
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ChatContext(id={self.id}, message_id={self.message_id})>"


# Data classes for use in application logic
@dataclass
class WebPageData:
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    source_url: Optional[str] = None
    last_crawled: datetime = None
    
    @classmethod
    def from_orm(cls, web_page: WebPage):
        return cls(
            url=web_page.url,
            title=web_page.title,
            source_url=web_page.source_url,
            last_crawled=web_page.last_crawled
        )


@dataclass
class TextChunkData:
    content: str
    web_page_id: int
    chunk_index: int
    embedding: Optional[List[float]] = None
    metadata: Optional[dict] = None
    
    @classmethod
    def from_orm(cls, chunk: TextChunk):
        return cls(
            content=chunk.content,
            web_page_id=chunk.web_page_id,
            chunk_index=chunk.chunk_index,
            embedding=chunk.embedding.tolist() if chunk.embedding else None,
            metadata=chunk.metadata
        )