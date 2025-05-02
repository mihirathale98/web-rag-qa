import openai
from typing import List, Dict, Any, Optional, Union
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import Config
from .model import LLMClient, ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIClient(LLMClient):
    """OpenAI API client implementation"""
    
    def __init__(self, api_key: str = None):
        """Initialize the OpenAI client"""
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_config = Config.get_model_config(Config.ModelProvider.OPENAI)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError))
    )
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to OpenAI"""
        try:
            # Prepare request parameters
            params = request.to_dict()
            
            # Apply default values if not provided
            if not params.get("model"):
                params["model"] = self.model_config.get("model")
            if not params.get("temperature") and params.get("temperature") != 0:
                params["temperature"] = self.model_config.get("temperature")
            if not params.get("max_tokens"):
                params["max_tokens"] = self.model_config.get("max_tokens")
            
            # Make the API call
            response = self.client.chat.completions.create(**params)
            
            # Extract the response content
            message = response.choices[0].message
            content = message.content or ""
            
            # Prepare tool calls if any
            tool_calls = None
            if message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            # Prepare usage info
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            # Create response object
            return ChatResponse(
                content=content,
                model=response.model,
                usage=usage,
                role="assistant",
                finish_reason=response.choices[0].finish_reason,
                tool_calls=tool_calls
            )
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit error with OpenAI API: {str(e)}")
            raise
        except openai.APIConnectionError as e:
            logger.warning(f"Connection error with OpenAI API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error with OpenAI chat API: {str(e)}")
            return ChatResponse(
                content=f"Error: {str(e)}",
                model=params.get("model", "unknown"),
                usage={}
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError))
    )
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using OpenAI"""
        try:
            # Prepare request parameters
            params = request.to_dict()
            
            # Apply default values if not provided
            if not params.get("model"):
                params["model"] = self.model_config.get("embedding_model")
            
            # Make the API call
            response = self.client.embeddings.create(**params)
            
            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            
            # Prepare usage info
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            # Create response object
            return EmbeddingResponse(
                embeddings=embeddings,
                model=response.model,
                usage=usage
            )
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit error with OpenAI embeddings API: {str(e)}")
            raise
        except openai.APIConnectionError as e:
            logger.warning(f"Connection error with OpenAI embeddings API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error with OpenAI embeddings API: {str(e)}")
            raise