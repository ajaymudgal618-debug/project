"""Unit tests for malformed JSON recovery, position-swap, and audit logging."""

import tempfile
import unittest
from pathlib import Path

from judge_pipeline.config import JudgeConfig
from judge_pipeline.judge import judge_pair
from judge_pipeline.parser import parse_verdict


class TestJudgePipeline(unittest.TestCase):
    def test_malformed_json_recovery(self) -> None:
        # Fenced + trailing comma
        raw1 = '```json\n{"scores_a":{"correctness":5,},"scores_b":{"correctness":2},"winner":"A","rationale":"ok",}\n```'
        v1 = parse_verdict(raw1)
        self.assertEqual(v1.winner, "A")
        self.assertEqual(v1.parse_status, "brace_repaired")

        # Regex fallback
        raw2 = 'Verdict: "winner": "B", "rationale": "more accurate"'
        v2 = parse_verdict(raw2)
        self.assertEqual(v2.winner, "B")
        self.assertEqual(v2.parse_status, "regex_fallback")

    def test_position_swap_and_audit_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "audit.jsonl"
            cfg = JudgeConfig(audit_log_path=str(log_file))
            v = judge_pair(
                case_id="t1",
                case_input="What is 2+2?",
                system_prompt="Be accurate.",
                output_a="4",
                output_b="5",
                expected_output="4",
                cfg=cfg,
                swap_positions=True,
            )
            self.assertEqual(v.final_winner, "A")
            self.assertFalse(v.position_flipped)
            lines = log_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)  # AB and BA logged


if __name__ == "__main__":
    unittest.main()
