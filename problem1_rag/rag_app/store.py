"""SQLite + NumPy low-cost embedded vector store with metadata filtering."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    source_path: str
    source_type: str
    topic: str
    score: float


class SQLiteVectorStore:
    """Single-file SQLite vector store with NumPy cosine similarity search."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    ingested_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_topic ON chunks(topic)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")

    def get_document_hash(self, doc_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return row[0] if row else None

    def upsert_document(
        self,
        doc_id: str,
        source_path: str,
        source_type: str,
        topic: str,
        content_hash: str,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> Tuple[int, int]:
        """Idempotently upsert a document and its chunks.

        Returns (inserted_count, skipped_count).
        """
        existing_hash = self.get_document_hash(doc_id)
        if existing_hash == content_hash:
            return (0, len(chunks))

        now = time.time()
        with self._connect() as conn:
            if existing_hash is not None:
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

            conn.execute(
                """
                INSERT INTO documents (doc_id, source_path, source_type, topic, content_hash, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    source_type = excluded.source_type,
                    topic = excluded.topic,
                    content_hash = excluded.content_hash,
                    ingested_at = excluded.ingested_at
                """,
                (doc_id, source_path, source_type, topic, content_hash, now),
            )

            inserted = 0
            for idx, (ch, emb) in enumerate(zip(chunks, embeddings)):
                emb_bytes = emb.astype(np.float32).tobytes()
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO chunks
                    (chunk_id, doc_id, chunk_index, text, source_path, source_type, topic, metadata_json, embedding, dim)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ch["chunk_id"],
                        doc_id,
                        idx,
                        ch["text"],
                        source_path,
                        source_type,
                        topic,
                        json.dumps(ch.get("metadata", {})),
                        emb_bytes,
                        int(emb.shape[0]),
                    ),
                )
                inserted += cur.rowcount
        return (inserted, len(chunks) - inserted)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        source_type: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        clauses = []
        params: list[Any] = []
        if source_type:
            clauses.append("source_type = ?")
            params.append(source_type.lower())
        if topic:
            clauses.append("topic = ?")
            params.append(topic.lower())

        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT chunk_id, doc_id, chunk_index, text, source_path, source_type, topic, embedding, dim
            FROM chunks
            {where_sql}
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        dim = rows[0][8]
        matrix = np.vstack([np.frombuffer(r[7], dtype=np.float32, count=dim) for r in rows])
        q = query_vec.astype(np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm
        scores = matrix @ q

        k = min(top_k, len(rows))
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: List[RetrievedChunk] = []
        for idx in top_indices:
            r = rows[int(idx)]
            results.append(
                RetrievedChunk(
                    chunk_id=r[0],
                    doc_id=r[1],
                    chunk_index=r[2],
                    text=r[3],
                    source_path=r[4],
                    source_type=r[5],
                    topic=r[6],
                    score=float(scores[int(idx)]),
                )
            )
        return results

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            by_type = conn.execute(
                "SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type"
            ).fetchall()
        db_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        return {
            "documents": doc_count,
            "vectors": chunk_count,
            "by_source_type": dict(by_type),
            "db_size_kb": round(db_bytes / 1024.0, 2),
        }
