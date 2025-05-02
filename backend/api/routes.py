from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
import asyncio
import uuid
import time
from typing import List, Dict, Any, Optional
import logging

from ..config import Config, ModelProvider
from ..database.db_client import db_client
from ..database.vector_store import VectorStore
from ..database.models import WebPage, TextChunk, ChatMessage, WebPageData, TextChunkData
from ..crawler.crawler import WebCrawler
from ..processor.cleaner import TextCleaner
from ..processor.chunker import TextChunker
from ..llm.model import ModelFactory, Message, ChatRequest as ModelChatRequest, EmbeddingRequest
from . import schemas

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Get dependencies
def get_db():
    with db_client.get_session() as session:
        yield session

def get_vector_store():
    return VectorStore()

def get_model_client(provider: Optional[ModelProvider] = None):
    provider = provider or Config.DEFAULT_MODEL_PROVIDER
    return ModelFactory.create_client(provider)

@router.post("/build_index", response_model=schemas.BuildIndexResponse)
async def build_index(
    request: schemas.BuildIndexRequest,
    background_tasks: BackgroundTasks,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Crawl URLs and build search index"""
    # Start processing in the background
    background_tasks.add_task(process_urls, [str(url) for url in request.urls], vector_store)
    
    return schemas.BuildIndexResponse(
        status="processing",
        message="URLs are being processed in the background"
    )

async def process_urls(urls: List[str], vector_store: VectorStore):
    """Process URLs: crawl, clean, chunk, and index"""
    # Create required components
    crawler = WebCrawler()
    cleaner = TextCleaner()
    chunker = TextChunker()
    llm_client = get_model_client()
    
    try:
        # Crawl the URLs
        crawl_result = await crawler.crawl_urls(urls)
        
        # Process each page
        for page_data in crawl_result.pages:
            if not page_data.content:
                continue
            
            # Clean the content
            cleaned_content = cleaner.clean(page_data.content)
            page_data.content = cleaned_content
            
            # Save the webpage to get an ID
            with db_client.get_session() as session:
                webpage = vector_store.add_webpage(page_data, session)
                
                # Chunk the content
                chunks = chunker.chunk_webpage(page_data)
                
                # Process chunks in batches to avoid overwhelming the embedding API
                batch_size = 10
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    
                    # Generate embeddings for the batch
                    texts = [chunk.content for chunk in batch]
                    embedding_request = EmbeddingRequest(text=texts)
                    embedding_response = await llm_client.embed(embedding_request)
                    
                    # Add embeddings to chunks and save
                    for j, chunk in enumerate(batch):
                        chunk.web_page_id = webpage.id
                        chunk.embedding = embedding_response.embeddings[j]
                        vector_store.add_chunk(chunk, session)
            
        logger.info(f"Successfully processed {len(crawl_result.pages)} pages")
        
    except Exception as e:
        logger.error(f"Error processing URLs: {str(e)}")

@router.post("/chat", response_model=schemas.ChatResponse)
async def chat(
    request: schemas.ChatRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    db: Session = Depends(get_db)
):
    """Chat with RAG-enhanced LLM"""
    try:
        # Extract the latest user message
        latest_user_message = None
        for msg in reversed(request.messages):
            if msg.role == schemas.MessageRole.USER:
                latest_user_message = msg
                break
        
        if not latest_user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Generate embedding for the query
        llm_client = get_model_client()
        embedding_request = EmbeddingRequest(text=latest_user_message.content)
        embedding_response = await llm_client.embed(embedding_request)
        query_embedding = embedding_response.embeddings[0]
        
        # Retrieve relevant chunks using vector search
        results = vector_store.similarity_search(query_embedding, top_k=5)
        
        # Convert results to text for context
        context_text = ""
        for chunk, score in results:
            # Add metadata about the source
            webpage = db.query(WebPage).filter(WebPage.id == chunk.web_page_id).first()
            source_info = f"Source: {webpage.title or 'Untitled'} ({webpage.url})\n"
            
            # Add the chunk content with separator
            context_text += f"{source_info}\n{chunk.content}\n\n---\n\n"
        
        # Create a system message with the context
        system_message = f"You are a helpful assistant with access to the following information. " \
                         f"Use this information to answer the user's question, but don't " \
                         f"mention the information directly unless relevant to the answer. " \
                         f"If the information doesn't help with the user's query, just respond " \
                         f"based on your general knowledge.\n\n{context_text}"
        
        # Prepare messages for the LLM
        messages = [Message(role="system", content=system_message)]
        
        # Add conversation history (excluding the system message if present)
        for msg in request.messages:
            if msg.role != schemas.MessageRole.SYSTEM:
                messages.append(Message(
                    role=msg.role.value,
                    content=msg.content,
                    name=msg.name,
                    function_call=msg.function_call
                ))
        
        # Create a model request
        model_request = ModelChatRequest(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools,
            tool_choice=request.tool_choice
        )
        
        # Get response from LLM
        model_response = await llm_client.chat(model_request)
        
        # Save to database if needed
        # This is optional but useful for tracking conversations
        conversation_id = str(uuid.uuid4())  # Generate new ID or extract from request if available
        
        # Create chat choice
        choice = schemas.ChatResponseChoice(
            index=0,
            message=schemas.MessageSchema(
                role=schemas.MessageRole.ASSISTANT,
                content=model_response.content
            ),
            finish_reason=model_response.finish_reason
        )
        
        # Prepare token usage
        usage = schemas.Usage(
            prompt_tokens=model_response.usage.get("prompt_tokens", 0),
            completion_tokens=model_response.usage.get("completion_tokens", 0),
            total_tokens=model_response.usage.get("total_tokens", 0)
        )
        
        # Create final response
        response = schemas.ChatResponse(
            id=f"chatcmpl-{str(uuid.uuid4())}",
            object="chat.completion",
            created=int(time.time()),
            model=model_response.model,
            choices=[choice],
            usage=usage
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))