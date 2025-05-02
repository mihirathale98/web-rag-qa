from fastapi import FastAPI, Depends, Body, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from typing import List, Dict, Any

from config import Config
from database.db_client import db_client
from api.routes import router
from api.schemas import ChatRequest, BuildIndexRequest

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION
)

# Configure CORS
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
@app.on_event("startup")
async def startup_event():
    try:
        # Setup database with pgvector extension
        db_client.setup()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")

# Include API router
app.include_router(router, prefix="/api")

# Recreate the original endpoints to maintain compatibility
@app.post("/build_index")
async def build_index_compat(urls: list = Body(...)):
    """Compatible endpoint for build_index"""
    try:
        # Convert to proper request object
        request = BuildIndexRequest(urls=urls)
        # Call the router endpoint
        response = await router.url_path_for("build_index")(request)
        return JSONResponse(content=jsonable_encoder({"status": "success"}), media_type="application/json")
    except Exception as e:
        logger.error(f"Error in build_index: {str(e)}")
        return JSONResponse(
            content=jsonable_encoder({"status": "error", "message": str(e)}),
            status_code=500,
            media_type="application/json"
        )

@app.post("/chat")
async def chat_compat(prompt: str = Body(...)):
    """Compatible endpoint for chat"""
    try:
        # Convert to proper request object
        request = ChatRequest(
            messages=[{"role": "user", "content": prompt}],
            model=Config.get_model_config().get("model")
        )
        # Call the router endpoint
        response = await router.url_path_for("chat")(request)
        return JSONResponse(content=jsonable_encoder({"response": response.choices[0].message.content}), media_type="application/json")
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return JSONResponse(
            content=jsonable_encoder({"response": f"Error: {str(e)}"}),
            status_code=500,
            media_type="application/json"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)