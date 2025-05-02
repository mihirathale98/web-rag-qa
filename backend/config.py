import os
from dotenv import load_dotenv
from enum import Enum
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    TOGETHER = "together"

class Config:
    # Database settings
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "ragchatbot")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    
    # Database connection string
    DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Vector settings
    EMBEDDING_DIMENSION = 1536  # Default for OpenAI ada-002
    
    # Crawler settings
    CRAWLER_MAX_DEPTH = 1  # 1-hop crawl
    CRAWLER_TIMEOUT = 10  # seconds
    CRAWLER_HEADERS = {
        "User-Agent": "RAGChatBot/1.0"
    }
    
    # Chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # LLM settings
    DEFAULT_MODEL_PROVIDER = ModelProvider.OPENAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    
    # Default model configurations
    MODEL_CONFIGS = {
        ModelProvider.OPENAI: {
            "model": "gpt-4o",
            "embedding_model": "text-embedding-ada-002",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        ModelProvider.ANTHROPIC: {
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        ModelProvider.GEMINI: {
            "model": "gemini-pro",
            "temperature": 0.7,
            "max_output_tokens": 1000,
        },
        ModelProvider.TOGETHER: {
            "model": "togethercomputer/llama-2-70b-chat",
            "temperature": 0.7,
            "max_tokens": 1000,
        }
    }
    
    # API settings
    API_TITLE = "RAG ChatBot API"
    API_DESCRIPTION = "API for RAG-based chatbot with web crawling capabilities"
    API_VERSION = "0.1.0"
    
    @classmethod
    def get_model_config(cls, provider: Optional[ModelProvider] = None) -> Dict[str, Any]:
        """Get the configuration for a specific model provider"""
        provider = provider or cls.DEFAULT_MODEL_PROVIDER
        return cls.MODEL_CONFIGS.get(provider, cls.MODEL_CONFIGS[cls.DEFAULT_MODEL_PROVIDER])