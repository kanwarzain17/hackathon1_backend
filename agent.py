#!/usr/bin/env python3
"""
Book Content RAG Agent
- Cohere embeddings (embed-english-v3.0)
- Qdrant vector search
- Cohere for text generation
- Strictly answers ONLY from retrieved book content
"""

import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# External libs (only used if configured)
import cohere
from qdrant_client import QdrantClient

# --------------------------------------------------
# ENV & LOGGING
# --------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("book-rag")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "book_embeddings")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    logger.warning("COHERE_API_KEY is missing — agent will run in fallback mode")

if not QDRANT_URL:
    logger.warning("QDRANT_URL not set — using local Qdrant or disabling vector search")

# --------------------------------------------------
# BOOK CONTENT AGENT
# --------------------------------------------------
class BookContentAgent:
    def __init__(self):
        self.cohere_client = None
        self.qdrant = None

        logger.info("BookContentAgent initialized")
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize external services only if config exists"""

        # Cohere
        if COHERE_API_KEY and self.cohere_client is None:
            try:
                logger.info("Initializing Cohere client...")
                self.cohere_client = cohere.Client(COHERE_API_KEY)
                logger.info("Cohere client initialized")
            except Exception:
                logger.exception("Failed to initialize Cohere client")
                self.cohere_client = None

        # Qdrant
        if self.qdrant is None:
            try:
                logger.info("Initializing Qdrant client...")
                if QDRANT_URL:
                    self.qdrant = QdrantClient(
                        url=QDRANT_URL,
                        api_key=QDRANT_API_KEY,
                        timeout=30
                    )
                else:
                    self.qdrant = QdrantClient(
                        host=os.getenv("QDRANT_HOST", "localhost"),
                        port=int(os.getenv("QDRANT_PORT", 6333))
                    )

                self.qdrant.get_collections()
                logger.info(f"Connected to Qdrant collection: {QDRANT_COLLECTION}")

            except Exception:
                logger.exception("Failed to initialize Qdrant")
                self.qdrant = None

    # --------------------------------------------------
    # QUERY
    # --------------------------------------------------
    def query(self, user_input: str) -> str:
        if not user_input or not user_input.strip():
            return "Query cannot be empty."

        # Greeting shortcut
        greetings = {"hello", "hi", "hey", "good morning", "good evening", "good afternoon"}
        if user_input.lower().strip() in greetings:
            return (
                "Hello! I'm your Book Assistant. "
                "Ask me anything about ROS 2, humanoid robotics, or the course material."
            )

        # Off-topic guard
        off_topic_keywords = [
            "weather", "joke", "news", "sports", "movie", "celebrity",
            "crypto", "recipe", "food", "travel", "music", "song",
            "health", "medical", "exercise", "diet"
        ]

        if any(k in user_input.lower() for k in off_topic_keywords):
            return (
                "I can only answer questions related to the book content. "
                "Please ask about robotics, ROS 2, or the course material."
            )

        # Hard stop if AI not configured
        if not self.cohere_client:
            return (
                "⚠️ AI services are not configured.\n\n"
                "Please set COHERE_API_KEY to enable question answering."
            )

        try:
            # --------------------------------------------------
            # EMBEDDING
            # --------------------------------------------------
            embed_response = self.cohere_client.embed(
                texts=[user_input],
                model="embed-english-v3.0",
                input_type="search_query"
            )

            query_vector = embed_response.embeddings[0]

            # --------------------------------------------------
            # VECTOR SEARCH
            # --------------------------------------------------
            retrieved_content: List[Dict[str, Any]] = []

            if self.qdrant:
                hits = self.qdrant.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=query_vector,
                    limit=5,
                    with_payload=True
                )

                for hit in hits:
                    score = float(hit.score or 0.0)
                    if score >= 0.3:
                        text = (hit.payload or {}).get("content", "")
                        if text.strip():
                            retrieved_content.append(
                                {"content": text[:800], "score": score}
                            )

            logger.info(f"Retrieved {len(retrieved_content)} chunks")

            if not retrieved_content:
                return (
                    "No relevant book content found. "
                    "Try asking about ROS 2 or humanoid robotics."
                )

            # --------------------------------------------------
            # RAG PROMPT
            # --------------------------------------------------
            context = "\n\n---\n\n".join(
                f"[Excerpt {i+1}]:\n{item['content']}"
                for i, item in enumerate(retrieved_content)
            )

            rag_prompt = f"""
You are a helpful assistant that answers questions ONLY using the provided book excerpts below.

BOOK EXCERPTS:
{context}

USER QUESTION:
{user_input}

INSTRUCTIONS:
- Answer ONLY using information from the book excerpts.
- Do NOT use outside knowledge.
- Be concise and accurate.

ANSWER:
"""

            # --------------------------------------------------
            # GENERATION
            # --------------------------------------------------
            response = self.cohere_client.chat(
                model="command-a-03-2025",
                message=rag_prompt,
                temperature=0.2,
                max_tokens=500
            )

            answer = response.text.strip() if response.text else ""
            return answer or "No response generated. Please rephrase your question."

        except Exception:
            logger.exception("RAG query failed")
            return (
                "Sorry, I encountered an internal error while processing your request."
            )

    def reset(self):
        """Reset agent state if needed"""
        pass


# --------------------------------------------------
# LOCAL TEST
# --------------------------------------------------
if __name__ == "__main__":
    agent = BookContentAgent()
    print(agent.query("What is ROS 2?"))
