"""Unit and integration tests for Problem 1 RAG application."""

import tempfile
import unittest
from pathlib import Path

from rag_app.config import RagConfig
from rag_app.ingest import ingest_directory
from rag_app.qa import NO_CONTEXT_ANSWER, answer_question
from rag_app.store import SQLiteVectorStore


class TestRagPipeline(unittest.TestCase):
    def test_idempotent_ingest_and_metadata_filter(self) -> None:
        corpus = str(Path(__file__).resolve().parent.parent / "rag_app" / "eval" / "corpus")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = RagConfig(db_path=f"{tmp}/test.db", log_path=f"{tmp}/q.jsonl")
            r1 = ingest_directory(corpus, cfg=cfg)
            self.assertGreater(r1["chunks_inserted"], 0)
            v1 = r1["store_stats"]["vectors"]

            r2 = ingest_directory(corpus, cfg=cfg)
            self.assertEqual(r2["chunks_inserted"], 0)
            self.assertEqual(r2["store_stats"]["vectors"], v1)

            # Metadata filter: only pdf chunks
            res_pdf = answer_question(
                "How much RAM does truncating embeddings save?",
                cfg=cfg,
                source_type="pdf",
            )
            self.assertFalse(res_pdf.no_relevant_context)
            self.assertTrue(all(c["source_type"] == "pdf" for c in res_pdf.retrieved_chunks))

    def test_no_relevant_context_refusal(self) -> None:
        corpus = str(Path(__file__).resolve().parent.parent / "rag_app" / "eval" / "corpus")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = RagConfig(db_path=f"{tmp}/test.db", log_path=f"{tmp}/q.jsonl")
            ingest_directory(corpus, cfg=cfg)
            res = answer_question("Who won the 1998 FIFA World Cup final?", cfg=cfg)
            self.assertTrue(res.no_relevant_context)
            self.assertEqual(res.answer, NO_CONTEXT_ANSWER)
            self.assertEqual(res.chunk_count, 0)


if __name__ == "__main__":
    unittest.main()
