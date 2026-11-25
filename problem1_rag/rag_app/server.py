"""FastAPI HTTP server exposing the Cost-Efficient RAG service."""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .config import get_config
from .ingest import ingest_directory
from .qa import answer_question
from .store import SQLiteVectorStore

app = FastAPI(title="Cost-Efficient RAG Service", version="1.0.0")


class IngestRequest(BaseModel):
    corpus_dir: str
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    source_type: Optional[str] = None
    topic: Optional[str] = None
    min_score: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    cfg = get_config()
    return SQLiteVectorStore(cfg.db_path).stats()


@app.post("/ingest")
def ingest(req: IngestRequest):
    cfg = get_config()
    return ingest_directory(
        req.corpus_dir, cfg=cfg, chunk_size=req.chunk_size, overlap=req.overlap
    )


@app.post("/query")
def query(req: QueryRequest):
    cfg = get_config()
    res = answer_question(
        req.question,
        cfg=cfg,
        top_k=req.top_k,
        source_type=req.source_type,
        topic=req.topic,
        min_score=req.min_score,
    )
    return asdict(res)
