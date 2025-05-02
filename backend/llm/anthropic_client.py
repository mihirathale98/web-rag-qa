import anthropic
from typing import List, Dict, Any, Optional, Union
import logging
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import Config
from .model import LLMClient, ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, Message

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnthropicClient(LLMClient):
    """Anthropic API client implementation"""
    
    def __init__(self, api_key: str = None):
        """Initialize the Anthropic client"""
        self.api_key = api_key or Config.ANTHROPIC_API_KEY
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model_config = Config.get_model_config(Config.ModelProvider.ANTHROPIC)
        
        # For embedding, we'll fall back to OpenAI since Anthropic doesn't have an embeddings API yet
        from .openai_client import OpenAIClient
        self.embedding_client = OpenAIClient()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            anthropic.RateLimitError, 
            anthropic.APIConnectionError,
            anthropic.APIStatusError
        ))
    )
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to Anthropic"""
        try:
            # Convert OpenAI-style messages to Anthropic format
            system_message = None
            messages = []
            
            for msg in request.messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    messages.append({"role": msg.role, "content": msg.content})
            
            # Prepare request parameters
            params = {
                "messages": messages,
                "model": request.model or self.model_config.get("model"),
                "max_tokens": request.max_tokens or self.model_config.get("max_tokens"),
                "temperature": request.temperature if request.temperature is not None else self.model_config.get("temperature")
            }
            
            # Add system message if provided
            if system_message:
                params["system"] = system_message
            
            # Convert tools to Anthropic tools format if provided
            if request.tools:
                # Anthropic has a different tool format than OpenAI
                anthropic_tools = []
                for tool in request.tools:
                    if tool.get("type") == "function":
                        anthropic_tools.append({
                            "name": tool["function"]["name"],
                            "description": tool["function"].get("description", ""),
                            "input_schema": tool["function"]["parameters"]
                        })
                
                if anthropic_tools:
                    params["tools"] = anthropic_tools
            
            # Make the API call
            response = self.client.messages.create(**params)
            
            # Extract tool calls if any
            tool_calls = None
            if response.content and any(block.get("type") == "tool_use" for block in response.content):
                tool_calls = []
                for block in response.content:
                    if block.get("type") == "tool_use":
                        tool_use = block["tool_use"]
                        tool_calls.append({
                            "id": tool_use.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tool_use["name"],
                                "arguments": json.dumps(tool_use["input"])
                            }
                        })
            
            # Extract the text content
            content = ""
            for block in response.content:
                if block.get("type") == "text":
                    content = block["text"]
                    break
            
            # Create response object
            return ChatResponse(
                content=content,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                role="assistant",
                finish_reason=response.stop_reason,
                tool_calls=tool_calls
            )
            
        except anthropic.RateLimitError as e:
            logger.warning(f"Rate limit error with Anthropic API: {str(e)}")
            raise
        except anthropic.APIConnectionError as e:
            logger.warning(f"Connection error with Anthropic API: {str(e)}")
            raise
        except anthropic.APIStatusError as e:
            logger.warning(f"Status error with Anthropic API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error with Anthropic API: {str(e)}")
            return ChatResponse(
                content=f"Error: {str(e)}",
                model=params.get("model", "unknown"),
                usage={}
            )
    
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using fallback to OpenAI"""
        # Anthropic doesn't have an embedding API yet, so we use OpenAI
        logger.info("Using OpenAI for embeddings as Anthropic doesn't provide embedding API")
        return await self.embedding_client.embed(request)