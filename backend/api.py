import logging
from contextlib import asynccontextmanager
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.rag_core import ConversationalRAG


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system

    print("Initializing RAG System on startup...")

    try:
        rag_system = ConversationalRAG()
        rag_system.initialize_chain()
        print("RAG System initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize RAG system: {e}")
        rag_system = None

    yield

    print("Shutting down API.")


app = FastAPI(
    title="Conversational RAG API",
    description="An API for interacting with the RAG chatbot.",
    version="1.0.0",
    lifespan=lifespan
)


class ChatRequest(BaseModel):
    question: str
    chat_history: List[Dict]


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    type: str


@app.get("/")
async def root():
    return {
        "message": "Enterprise Conversational AI API is running"
    }


@app.get("/status")
async def get_status():
    if rag_system:
        return {
            "rag_initialized": rag_system.rag_initialized
        }

    return {
        "rag_initialized": False
    }


@app.post("/process-documents")
async def process_documents():
    if not rag_system:
        raise HTTPException(
            status_code=500,
            detail="RAG system not available."
        )

    try:
        logger.info("API call received to process documents.")

        chunks_added = rag_system.load_and_process_documents()

        if chunks_added > 0:
            rag_system.initialize_chain()

        message = (
            f"Processing complete. Added {chunks_added} new document chunks."
            if chunks_added > 0
            else "Processing complete. No new documents found."
        )

        return {
            "message": message,
            "chunks_added": chunks_added
        }

    except Exception as e:
        logger.exception(f"Error processing documents: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not initialized."
        )

    try:
        response = rag_system.query(
            request.question,
            request.chat_history
        )

        return response

    except Exception as e:
        logger.exception(f"Chat error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )