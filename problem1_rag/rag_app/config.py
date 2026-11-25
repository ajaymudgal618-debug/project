"""Environment-driven configuration for the Cost-Efficient RAG service."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RagConfig:
    db_path: str = os.getenv("RAG_DB_PATH", "data/rag_store.db")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.16"))
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local_tfidf")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    llm_provider: str = os.getenv("LLM_PROVIDER", "local_extractive")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_embed_model: str = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
    gemini_chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    log_path: str = os.getenv("RAG_LOG_PATH", "logs/query_logs.jsonl")


def get_config() -> RagConfig:
    return RagConfig(
        db_path=os.getenv("RAG_DB_PATH", "data/rag_store.db"),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        min_score=float(os.getenv("RAG_MIN_SCORE", "0.16")),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local_tfidf"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
        llm_provider=os.getenv("LLM_PROVIDER", "local_extractive"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004"),
        gemini_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
        log_path=os.getenv("RAG_LOG_PATH", "logs/query_logs.jsonl"),
    )
