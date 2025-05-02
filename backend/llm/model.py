from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import json

from ..config import Config, ModelProvider

@dataclass
class Message:
    """Chat message format compatible with OpenAI message structure"""
    role: str  # "system", "user", "assistant", "function"
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.function_call:
            result["function_call"] = self.function_call
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a Message from a dictionary"""
        return cls(
            role=data.get("role"),
            content=data.get("content", ""),
            name=data.get("name"),
            function_call=data.get("function_call")
        )


@dataclass
class ChatRequest:
    """Chat request structure compatible with multiple model providers"""
    messages: List[Message]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests"""
        result = {
            "messages": [m.to_dict() for m in self.messages]
        }
        if self.model:
            result["model"] = self.model
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.max_tokens:
            result["max_tokens"] = self.max_tokens
        if self.tools:
            result["tools"] = self.tools
        if self.tool_choice:
            result["tool_choice"] = self.tool_choice
        return result


@dataclass
class ChatResponse:
    """Standardized chat response structure"""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    role: str = "assistant"
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def to_message(self) -> Message:
        """Convert to a Message object"""
        return Message(
            role=self.role,
            content=self.content,
            function_call=None  # We handle tool_calls separately
        )


@dataclass
class EmbeddingRequest:
    """Embedding request structure"""
    text: Union[str, List[str]]
    model: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests"""
        result = {
            "input": self.text
        }
        if self.model:
            result["model"] = self.model
        return result


@dataclass
class EmbeddingResponse:
    """Standardized embedding response structure"""
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int] = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to the LLM"""
        pass
    
    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for text"""
        pass


class ModelFactory:
    """Factory for creating LLM clients"""
    
    @staticmethod
    def create_client(provider: ModelProvider) -> LLMClient:
        """Create a client for the specified provider"""
        if provider == ModelProvider.OPENAI:
            from .openai_client import OpenAIClient
            return OpenAIClient()
        elif provider == ModelProvider.ANTHROPIC:
            from .anthropic_client import AnthropicClient
            return AnthropicClient()
        elif provider == ModelProvider.GEMINI:
            from .gemini_client import GeminiClient
            return GeminiClient()
        elif provider == ModelProvider.TOGETHER:
            from .together_client import TogetherClient
            return TogetherClient()
        else:
            raise ValueError(f"Unsupported model provider: {provider}")