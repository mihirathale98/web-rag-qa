from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator

from ..config import Config
from .models import Base

class DatabaseClient:
    """Database connection manager for PostgreSQL with pgvector extension"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or Config.DB_URL
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables if they don't exist"""
        Base.metadata.create_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with context management"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def init_pgvector(self):
        """Initialize pgvector extension if it doesn't exist"""
        with self.engine.connect() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.commit()
    
    def setup(self):
        """Complete database setup"""
        self.init_pgvector()
        self.create_tables()
        return self


# Global instance for use throughout the application
db_client = DatabaseClient()