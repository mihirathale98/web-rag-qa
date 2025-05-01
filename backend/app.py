from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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

@app.post("/build_index")
async def build_index(urls: list = Body(...)):
    # code to build index
    return JSONResponse(content=jsonable_encoder({"status": "success"}), media_type="application/json")

@app.post("/chat")
async def chat(prompt: str = Body(...)):
    # code to chat with ragchatbot
    return JSONResponse(content=jsonable_encoder({"response": prompt}), media_type="application/json")
