"""Grounded QA service with no-relevant-context handling, citations, and logging."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .config import RagConfig, get_config
from .embeddings import get_embedder
from .store import RetrievedChunk, SQLiteVectorStore

NO_CONTEXT_ANSWER = (
    "I don't have enough relevant context in the indexed corpus to answer that question."
)


@dataclass
class QueryResult:
    question: str
    answer: str
    citations: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    no_relevant_context: bool
    retrieval_ms: float
    total_ms: float
    chunk_count: int
    prompt_tokens: int
    completion_tokens: int
    llm_provider: str


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _local_extractive_answer(question: str, chunks: List[RetrievedChunk]) -> tuple[str, int, int]:
    """Deterministic grounded answer synthesis for offline/cost-free runs."""
    q_terms = set(re.findall(r"[a-z0-9_]+", question.lower()))
    scored_sents: list[tuple[float, str, str]] = []
    for ch in chunks:
        for sent in re.split(r"(?<=[.!?])\s+", ch.text.strip()):
            s_clean = sent.strip()
            if len(s_clean) < 15:
                continue
            s_terms = set(re.findall(r"[a-z0-9_]+", s_clean.lower()))
            overlap = len(q_terms & s_terms)
            score = overlap + ch.score * 2.0
            scored_sents.append((score, s_clean, ch.chunk_id))

    scored_sents.sort(key=lambda x: -x[0])
    chosen = scored_sents[:3]
    if not chosen:
        top = chunks[0]
        ans = f"{top.text[:280].strip()} [{top.chunk_id}]"
    else:
        ans = " ".join(f"{s} [{cid}]" for _, s, cid in chosen)

    prompt_toks = sum(_estimate_tokens(c.text) for c in chunks) + _estimate_tokens(question)
    comp_toks = _estimate_tokens(ans)
    return ans, prompt_toks, comp_toks


def _openai_answer(
    question: str, chunks: List[RetrievedChunk], cfg: RagConfig
) -> tuple[str, int, int]:
    if not cfg.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
    context_block = "\n\n".join(f"[{c.chunk_id}] ({c.source_path}): {c.text}" for c in chunks)
    system_prompt = (
        "You are a grounded QA assistant. Answer ONLY using the provided context chunks. "
        "Cite every factual claim inline using [chunk_id]. If the context does not contain "
        "the answer, reply exactly: " + NO_CONTEXT_ANSWER
    )
    payload = {
        "model": cfg.openai_chat_model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
        ],
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.openai_api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ans = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    return ans, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def _gemini_answer(
    question: str, chunks: List[RetrievedChunk], cfg: RagConfig
) -> tuple[str, int, int]:
    if not cfg.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
    context_block = "\n\n".join(f"[{c.chunk_id}] ({c.source_path}): {c.text}" for c in chunks)
    prompt = (
        "Answer ONLY using the provided context chunks. Cite every claim inline using [chunk_id]. "
        f"If the context is insufficient, reply exactly: {NO_CONTEXT_ANSWER}\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.gemini_chat_model}:generateContent?key={cfg.gemini_api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ans = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    meta = data.get("usageMetadata", {})
    return (
        ans,
        int(meta.get("promptTokenCount", 0)),
        int(meta.get("candidatesTokenCount", 0)),
    )


def _append_log(log_path: str, record: Dict[str, Any]) -> None:
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def answer_question(
    question: str,
    cfg: Optional[RagConfig] = None,
    top_k: Optional[int] = None,
    source_type: Optional[str] = None,
    topic: Optional[str] = None,
    min_score: Optional[float] = None,
) -> QueryResult:
    cfg = cfg or get_config()
    k = top_k if top_k is not None else cfg.top_k
    thresh = min_score if min_score is not None else cfg.min_score

    t0 = time.perf_counter()
    embedder = get_embedder(cfg)
    store = SQLiteVectorStore(cfg.db_path)

    q_vec = embedder.embed([question])[0]
    raw_chunks = store.search(q_vec, top_k=k, source_type=source_type, topic=topic)
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    relevant = [c for c in raw_chunks if c.score >= thresh]
    if not relevant:
        total_ms = (time.perf_counter() - t0) * 1000.0
        res = QueryResult(
            question=question,
            answer=NO_CONTEXT_ANSWER,
            citations=[],
            retrieved_chunks=[asdict(c) for c in raw_chunks],
            no_relevant_context=True,
            retrieval_ms=round(retrieval_ms, 3),
            total_ms=round(total_ms, 3),
            chunk_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            llm_provider=cfg.llm_provider,
        )
        _append_log(cfg.log_path, asdict(res))
        return res

    provider = cfg.llm_provider.lower()
    if provider == "openai":
        ans, p_tok, c_tok = _openai_answer(question, relevant, cfg)
    elif provider == "gemini":
        ans, p_tok, c_tok = _gemini_answer(question, relevant, cfg)
    else:
        ans, p_tok, c_tok = _local_extractive_answer(question, relevant)

    cited = [c.chunk_id for c in relevant if c.chunk_id in ans]
    if not cited and NO_CONTEXT_ANSWER not in ans:
        cited = [relevant[0].chunk_id]

    total_ms = (time.perf_counter() - t0) * 1000.0
    res = QueryResult(
        question=question,
        answer=ans,
        citations=cited,
        retrieved_chunks=[asdict(c) for c in relevant],
        no_relevant_context=NO_CONTEXT_ANSWER in ans,
        retrieval_ms=round(retrieval_ms, 3),
        total_ms=round(total_ms, 3),
        chunk_count=len(relevant),
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        llm_provider=cfg.llm_provider,
    )
    _append_log(cfg.log_path, asdict(res))
    return res
