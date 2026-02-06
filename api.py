"""
backend/api.py - FastAPI RAG Agent API
"""

import os
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
import logging

# ────────────────────────────────────────────────
# Debug prints at module level (runs on cold start / import)
# ────────────────────────────────────────────────
print("=== backend/api.py MODULE STARTED LOADING ===")
print(f"Current working directory: {os.getcwd()}")

# Load environment variables
load_dotenv()
print("Environment variables loaded via dotenv")

# Env var visibility (safe, no secrets leaked)
print(f"QDRANT_URL exists: {'QDRANT_URL' in os.environ}")
print(f"QDRANT_API_KEY exists: {'QDRANT_API_KEY' in os.environ}")
print(f"COHERE_API_KEY exists: {'COHERE_API_KEY' in os.environ}")
print(f"OPENROUTER_API_KEY exists: {'OPENROUTER_API_KEY' in os.environ}")

# Logging setup
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("api")
logger.info("Logging initialized")

# ────────────────────────────────────────────────
# Import Agent (FIXED: absolute import)
# ────────────────────────────────────────────────
try:
    from agent import BookContentAgent
    print("Successfully imported BookContentAgent")
except Exception as import_err:
    print(f"CRITICAL: Failed to import BookContentAgent: {import_err}")
    traceback.print_exc()
    raise

# ────────────────────────────────────────────────
# FastAPI App
# ────────────────────────────────────────────────
app = FastAPI(title="RAG Agent API")

# CORS (open for dev / hackathon)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────
# Pydantic Models
# ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class QueryResponse(BaseModel):
    response: str
    sources: List[str] = []
    session_id: Optional[str] = None
    timestamp: str
    status: str

# ────────────────────────────────────────────────
# Agent Manager
# ────────────────────────────────────────────────
class AgentManager:
    def __init__(self):
        self.agents: Dict[str, BookContentAgent] = {}
        print("AgentManager initialized")

    def get_agent(self, session_id: Optional[str] = None) -> BookContentAgent:
        key = session_id or "default"

        if key not in self.agents:
            print(f"Creating new agent for session: {key}")
            self.agents[key] = BookContentAgent()
        else:
            print(f"Reusing existing agent for session: {key}")

        return self.agents[key]

agent_manager = AgentManager()

# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "RAG Agent API is running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        agent = agent_manager.get_agent(request.session_id)
        response_text = agent.query(request.query)

        return QueryResponse(
            response=response_text,
            sources=[],
            session_id=request.session_id,
            timestamp=datetime.utcnow().isoformat(),
            status="success"
        )

    except Exception as e:
        print("ERROR in /query")
        traceback.print_exc()
        logger.error("Query failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

print("=== backend/api.py LOADED SUCCESSFULLY ===")
