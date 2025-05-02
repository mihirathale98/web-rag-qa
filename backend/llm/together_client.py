import together
from typing import List, Dict, Any, Optional, Union
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Config
from .model import LLMClient, ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TogetherClient(LLMClient):
    """Together AI API client implementation"""
    
    def __init__(self, api_key: str = None):
        """Initialize the Together AI client"""
        self.api_key = api_key or Config.TOGETHER_API_KEY
        together.api_key = self.api_key
        self.model_config = Config.get_model_config(Config.ModelProvider.TOGETHER)
        
        # For embedding, we'll fall back to OpenAI
        from .openai_client import OpenAIClient
        self.embedding_client = OpenAIClient()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to Together AI"""
        try:
            # Prepare request parameters from OpenAI-compatible format
            params = {
                "model": request.model or self.model_config.get("model"),
                "messages": [m.to_dict() for m in request.messages],
                "temperature": request.temperature if request.temperature is not None else self.model_config.get("temperature"),
                "max_tokens": request.max_tokens or self.model_config.get("max_tokens")
            }
            
            # Make the API call (Together AI has OpenAI-compatible API)
            response = together.Chat.create(**params)
            
            # Extract the response content
            content = response.choices[0].message.content
            
            # Prepare usage info
            usage = {}
            if hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }
            
            # Create response object
            return ChatResponse(
                content=content,
                model=response.model,
                usage=usage,
                role="assistant",
                finish_reason=response.choices[0].finish_reason,
                tool_calls=None  # Together AI models may not support tool calls in the same way
            )
            
        except Exception as e:
            logger.error(f"Error with Together AI API: {str(e)}")
            return ChatResponse(
                content=f"Error: {str(e)}",
                model=request.model or self.model_config.get("model", "unknown"),
                usage={}
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using Together AI or fallback"""
        try:
            # Check if Together AI has a suitable embedding model
            embedding_model = "togethercomputer/m2-embed-large"  # Example embedding model
            
            # Prepare request parameters
            params = {
                "model": embedding_model,
                "input": request.text
            }
            
            # Make the API call
            response = together.Embeddings.create(**params)
            
            # Extract embeddings
            embeddings = [data['embedding'] for data in response.data]
            
            # Create response object
            return EmbeddingResponse(
                embeddings=embeddings,
                model=embedding_model,
                usage={}  # Together AI might not provide detailed usage info
            )
            
        except Exception as e:
            logger.warning(f"Error with Together AI embeddings API: {str(e)}")
            logger.info("Falling back to OpenAI for embeddings")
            return await self.embedding_client.embed(request)