"""Three-layer RAG evaluation harness: Retrieval, Answer Quality, Cost & Latency."""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..config import RagConfig
from ..ingest import ingest_directory
from ..qa import NO_CONTEXT_ANSWER, answer_question


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _f1_score(pred: str, gold: str) -> float:
    p_toks = _tokens(pred)
    g_toks = _tokens(gold)
    if not p_toks or not g_toks:
        return 0.0
    common: Dict[str, int] = {}
    for t in p_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    prec = overlap / len(p_toks)
    rec = overlap / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def _exact_match(pred: str, gold: str) -> float:
    clean_pred = re.sub(r"\[[^\]]+\]", "", pred).strip().lower()
    return 1.0 if clean_pred == gold.strip().lower() else 0.0


def _is_chunk_relevant(chunk: Dict[str, Any], q_item: Dict[str, Any]) -> bool:
    if q_item["unanswerable"]:
        return False
    if chunk["doc_id"] != q_item["relevant_doc"]:
        return False
    text_low = chunk["text"].lower()
    kws = q_item.get("relevant_keywords", [])
    return any(kw.lower() in text_low for kw in kws) if kws else True


def _faithfulness_score(answer: str, chunks: List[Dict[str, Any]], no_ctx: bool) -> float:
    if no_ctx:
        return 1.0
    if not chunks:
        return 0.0
    ctx_tokens = set()
    for c in chunks:
        ctx_tokens.update(_tokens(c["text"]))
    ans_clean = re.sub(r"\[[^\]]+\]", "", answer)
    ans_toks = _tokens(ans_clean)
    if not ans_toks:
        return 0.0
    supported = sum(1 for t in ans_toks if t in ctx_tokens)
    return supported / len(ans_toks)


def _answer_relevance_score(question: str, answer: str, unanswerable: bool, no_ctx: bool) -> float:
    if unanswerable:
        return 1.0 if no_ctx else 0.0
    if no_ctx:
        return 0.0
    q_toks = set(_tokens(question))
    a_toks = set(_tokens(answer))
    if not q_toks:
        return 0.0
    return min(1.0, len(q_toks & a_toks) / max(1, len(q_toks) * 0.5))


def run_evaluation(
    corpus_dir: str,
    questions_path: str,
    db_path: str,
    top_k: int = 5,
    chunk_size: int = 500,
    overlap: int = 100,
    out_path: str = "eval_results.json",
) -> Dict[str, Any]:
    if os.path.exists(db_path):
        os.remove(db_path)

    cfg = RagConfig(
        db_path=db_path,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        top_k=top_k,
        min_score=0.16,
        log_path=os.path.join(os.path.dirname(out_path) or ".", "eval_query_logs.jsonl"),
    )

    ingest_1 = ingest_directory(corpus_dir, cfg=cfg)
    ingest_2 = ingest_directory(corpus_dir, cfg=cfg)  # Idempotency check

    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))

    recalls: List[float] = []
    mrrs: List[float] = []
    ndcgs: List[float] = []
    ctx_precs: List[float] = []

    faith_scores: List[float] = []
    rel_scores: List[float] = []
    em_scores: List[float] = []
    f1_scores: List[float] = []

    retrieval_latencies: List[float] = []
    total_latencies: List[float] = []
    per_question: List[Dict[str, Any]] = []

    for q_item in questions:
        res = answer_question(q_item["question"], cfg=cfg, top_k=top_k)
        retrieval_latencies.append(res.retrieval_ms)
        total_latencies.append(res.total_ms)

        if not q_item["unanswerable"]:
            rels = [1 if _is_chunk_relevant(c, q_item) else 0 for c in res.retrieved_chunks]
            hit = 1.0 if any(rels) else 0.0
            recalls.append(hit)

            rr = 0.0
            for idx, r in enumerate(rels, start=1):
                if r == 1:
                    rr = 1.0 / idx
                    break
            mrrs.append(rr)

            dcg = sum(r / math.log2(idx + 2) for idx, r in enumerate(rels))
            ideal_rels = sorted(rels, reverse=True)
            idcg = sum(r / math.log2(idx + 2) for idx, r in enumerate(ideal_rels))
            ndcgs.append((dcg / idcg) if idcg > 0 else 0.0)

            ctx_precs.append(sum(rels) / len(rels) if rels else 0.0)
            em_scores.append(_exact_match(res.answer, q_item["gold_answer"]))
            f1_scores.append(_f1_score(res.answer, q_item["gold_answer"]))

        faith_scores.append(_faithfulness_score(res.answer, res.retrieved_chunks, res.no_relevant_context))
        rel_scores.append(
            _answer_relevance_score(
                q_item["question"], res.answer, q_item["unanswerable"], res.no_relevant_context
            )
        )

        per_question.append(
            {
                "id": q_item["id"],
                "unanswerable": q_item["unanswerable"],
                "no_relevant_context": res.no_relevant_context,
                "retrieval_ms": res.retrieval_ms,
                "total_ms": res.total_ms,
                "chunk_count": res.chunk_count,
                "citations": res.citations,
            }
        )

    cost_comparison = [
        {
            "vectors": "100K",
            "sqlite_store_usd_mo": 6.00,
            "managed_db_usd_mo": 70.00,
            "savings_pct": "91.4%",
            "notes": "1 vCPU / 1GB RAM VM ($6/mo) vs 1x managed starter pod ($70/mo)",
        },
        {
            "vectors": "1M",
            "sqlite_store_usd_mo": 24.00,
            "managed_db_usd_mo": 140.00,
            "savings_pct": "82.9%",
            "notes": "2 vCPU / 4GB RAM VM + 10GB SSD ($24/mo) vs 2x RAM pods ($140/mo)",
        },
        {
            "vectors": "10M",
            "sqlite_store_usd_mo": 96.00,
            "managed_db_usd_mo": 420.00,
            "savings_pct": "77.1%",
            "notes": "4 vCPU / 16GB RAM VM + 50GB SSD ($96/mo) vs replicated 10M pod ($420/mo)",
        },
    ]

    summary = {
        "k": top_k,
        "num_questions": len(questions),
        "num_answerable": len(recalls),
        "num_unanswerable": len(questions) - len(recalls),
        "ingest_initial": ingest_1,
        "ingest_idempotent_rerun": ingest_2,
        "retrieval_metrics": {
            "recall_at_k": round(float(np.mean(recalls)), 4),
            "mrr": round(float(np.mean(mrrs)), 4),
            "ndcg_at_k": round(float(np.mean(ndcgs)), 4),
            "context_precision": round(float(np.mean(ctx_precs)), 4),
        },
        "answer_quality": {
            "faithfulness_lexical_grounding": round(float(np.mean(faith_scores)), 4),
            "answer_relevance": round(float(np.mean(rel_scores)), 4),
            "exact_match": round(float(np.mean(em_scores)), 4),
            "token_f1": round(float(np.mean(f1_scores)), 4),
            "unanswerable_refusal_rate": round(
                sum(1 for p in per_question if p["unanswerable"] and p["no_relevant_context"])
                / max(1, sum(1 for p in per_question if p["unanswerable"])),
                4,
            ),
        },
        "latency_ms": {
            "retrieval_p50": round(float(np.percentile(retrieval_latencies, 50)), 3),
            "retrieval_p95": round(float(np.percentile(retrieval_latencies, 95)), 3),
            "end_to_end_p50": round(float(np.percentile(total_latencies, 50)), 3),
            "end_to_end_p95": round(float(np.percentile(total_latencies, 95)), 3),
        },
        "cost_comparison": cost_comparison,
        "per_question": per_question,
    }

    Path(out_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    res = run_evaluation(
        corpus_dir=str(base / "corpus"),
        questions_path=str(base / "questions.json"),
        db_path=str(base.parent.parent / "data" / "eval_rag.db"),
        out_path=str(base.parent.parent / "eval_results.json"),
    )
    print(json.dumps({k: v for k, v in res.items() if k != "per_question"}, indent=2))
