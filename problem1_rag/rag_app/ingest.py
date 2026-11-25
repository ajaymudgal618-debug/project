"""Document ingestion pipeline for PDF, HTML, and Markdown files."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .config import RagConfig, get_config
from .embeddings import get_embedder
from .store import SQLiteVectorStore

SUPPORTED_EXTS = {".pdf", ".html", ".htm", ".md", ".markdown"}


def extract_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    if ext in {".html", ".htm"}:
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    return file_path.read_text(encoding="utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Prefer breaking at sentence or whitespace boundary in the last 20%
            window_start = start + int(chunk_size * 0.8)
            split_at = -1
            for sep in ("\n\n", ". ", "\n", " "):
                pos = text.rfind(sep, window_start, end)
                if pos != -1:
                    split_at = pos + len(sep)
                    break
            if split_at > start:
                end = split_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(start + 1, end - overlap)
    return chunks


def _infer_topic(file_path: Path, corpus_root: Path) -> str:
    rel = file_path.relative_to(corpus_root)
    if len(rel.parts) > 1:
        return rel.parts[0].lower()
    stem = file_path.stem.lower()
    return stem.split("_")[0]


def ingest_directory(
    corpus_dir: str,
    cfg: Optional[RagConfig] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = cfg or get_config()
    c_size = chunk_size if chunk_size is not None else cfg.chunk_size
    c_overlap = overlap if overlap is not None else cfg.chunk_overlap

    root = Path(corpus_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    embedder = get_embedder(cfg)
    store = SQLiteVectorStore(cfg.db_path)

    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )

    total_inserted = 0
    total_skipped = 0
    processed_docs = 0

    for fp in files:
        rel_path = str(fp.relative_to(root))
        raw_text = extract_text(fp)
        text = clean_text(raw_text)
        if not text:
            continue

        content_hash = hashlib.sha256(
            f"{rel_path}|{c_size}|{c_overlap}|{text}".encode("utf-8")
        ).hexdigest()
        doc_id = fp.stem
        source_type = fp.suffix.lower().lstrip(".")
        if source_type == "htm":
            source_type = "html"
        elif source_type == "markdown":
            source_type = "md"
        topic = _infer_topic(fp, root)

        pieces = chunk_text(text, chunk_size=c_size, overlap=c_overlap)
        chunk_records = []
        for idx, piece in enumerate(pieces):
            chunk_hash = hashlib.sha256(f"{doc_id}:{idx}:{piece}".encode("utf-8")).hexdigest()[:12]
            chunk_records.append(
                {
                    "chunk_id": f"{doc_id}#{idx}_{chunk_hash}",
                    "text": piece,
                    "metadata": {"source_path": rel_path, "chunk_index": idx},
                }
            )

        existing_hash = store.get_document_hash(doc_id)
        if existing_hash == content_hash:
            total_skipped += len(chunk_records)
            processed_docs += 1
            continue

        embeddings = embedder.embed([c["text"] for c in chunk_records])
        ins, skip = store.upsert_document(
            doc_id=doc_id,
            source_path=rel_path,
            source_type=source_type,
            topic=topic,
            content_hash=content_hash,
            chunks=chunk_records,
            embeddings=embeddings,
        )
        total_inserted += ins
        total_skipped += skip
        processed_docs += 1

    stats = store.stats()
    return {
        "files_scanned": len(files),
        "documents_processed": processed_docs,
        "chunks_inserted": total_inserted,
        "chunks_skipped": total_skipped,
        "chunk_size": c_size,
        "overlap": c_overlap,
        "embedding_model": embedder.model_name,
        "store_stats": stats,
    }
