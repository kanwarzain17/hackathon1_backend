"""
run_api.py
Local development launcher for FastAPI RAG Agent API
"""

import uvicorn
import argparse
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run the RAG Agent API server locally")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes"
    )

    args = parser.parse_args()
    host = args.host
    port = args.port

    print(f"Starting RAG Agent API server on {host}:{port}")
    print("Press Ctrl+C to stop the server")

    # Import the FastAPI app from api.py
    # Make sure api.py is in the same folder as run_api.py or adjust import accordingly
    import api

    uvicorn.run(
        "api:app",  # points to the FastAPI app in api.py
        host=host,
        port=port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
    