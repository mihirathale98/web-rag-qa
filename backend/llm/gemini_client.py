import google.generativeai as genai
from typing import List, Dict, Any, Optional, Union
import logging
import json
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Config
from .model import LLMClient, ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient(LLMClient):
    """Google Gemini API client implementation"""
    
    def __init__(self, api_key: str = None):
        """Initialize the Gemini client"""
        self.api_key = api_key or Config.GEMINI_API_KEY
        genai.configure(api_key=self.api_key)
        self.model_config = Config.get_model_config(Config.ModelProvider.GEMINI)
        
        # For embedding, we'll fall back to OpenAI
        from .openai_client import OpenAIClient
        self.embedding_client = OpenAIClient()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to Gemini"""
        try:
            # Convert OpenAI-style messages to Gemini format
            system_prompt = None
            gemini_messages = []
            
            for msg in request.messages:
                if msg.role == "system":
                    system_prompt = msg.content
                elif msg.role == "user":
                    gemini_messages.append({"role": "user", "parts": [msg.content]})
                elif msg.role == "assistant":
                    gemini_messages.append({"role": "model", "parts": [msg.content]})
                # Gemini doesn't support function messages directly, may need custom handling
            
            # Get model name from config if not provided
            model_name = request.model or self.model_config.get("model")
            
            # Create a Gemini model
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": request.temperature if request.temperature is not None else self.model_config.get("temperature"),
                    "max_output_tokens": request.max_tokens or self.model_config.get("max_output_tokens"),
                }
            )
            
            # If we have a system prompt, prepend it to the conversation
            if system_prompt:
                # Gemini doesn't have a dedicated system message, so we add it as a user message at the start
                system_msg = {"role": "user", "parts": [f"System instructions: {system_prompt}"]}
                model_response = {"role": "model", "parts": ["I'll follow these instructions."]}
                
                if len(gemini_messages) > 0 and gemini_messages[0]["role"] == "user":
                    # If first message is from user, we insert the system prompt and acknowledgement before it
                    gemini_messages = [system_msg, model_response] + gemini_messages
                else:
                    # Otherwise just prepend
                    gemini_messages = [system_msg, model_response] + gemini_messages
            
            # Create a chat session and send the messages
            chat = model.start_chat(history=gemini_messages[:-1] if gemini_messages else [])
            
            # Send the latest message if there is one
            if gemini_messages:
                response = chat.send_message(gemini_messages[-1]["parts"][0])
            else:
                # If no messages, send an empty prompt
                response = chat.send_message("")
            
            # Create response object
            return ChatResponse(
                content=response.text,
                model=model_name,
                usage={},  # Gemini doesn't provide detailed token usage
                role="assistant",
                finish_reason=None,  # Gemini doesn't provide finish reason
                tool_calls=None  # Gemini's function calling would need custom handling
            )
            
        except Exception as e:
            logger.error(f"Error with Gemini API: {str(e)}")
            return ChatResponse(
                content=f"Error: {str(e)}",
                model=request.model or self.model_config.get("model", "unknown"),
                usage={}
            )
    
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using fallback to OpenAI"""
        # Gemini doesn't have a general embedding API yet, so we use OpenAI
        logger.info("Using OpenAI for embeddings as Gemini doesn't provide a compatible embedding API")
        return await self.embedding_client.embed(request)