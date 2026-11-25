"""Suite runner, A/B comparison, bias measurement, and judge validation CLI."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from .config import JudgeConfig, get_judge_config
from .judge import PairwiseVerdict, judge_pair


def load_suite(path: str) -> Dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return json.loads(raw)
    return yaml.safe_load(raw)


def _cohens_kappa(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    if n == 0:
        return 0.0
    p_o = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    p_e = sum(
        (sum(1 for a in y_true if a == lbl) / n) * (sum(1 for b in y_pred if b == lbl) / n)
        for lbl in labels
    )
    if p_e >= 1.0:
        return 1.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


def run_evaluation_suite(
    suite_path: str,
    out_path: str = "judge_report.json",
    cfg: JudgeConfig | None = None,
) -> Dict[str, Any]:
    cfg = cfg or get_judge_config()
    suite = load_suite(suite_path)
    cases = suite["cases"]
    probes = suite.get("adversarial_probes", [])

    # 1. Measure unmitigated position bias (single order + primacy bonus) vs mitigated (both orders)
    unmitigated_flips = 0
    mitigated_verdicts: List[PairwiseVerdict] = []

    for c in cases:
        v_raw_ab = judge_pair(
            case_id=c["id"],
            case_input=c["input"],
            system_prompt=c["system_prompt"],
            output_a=c["output_a"],
            output_b=c["output_b"],
            expected_output=c.get("expected_output", ""),
            cfg=cfg,
            swap_positions=False,
            inject_position_bias=True,
        )
        v_raw_ba = judge_pair(
            case_id=c["id"],
            case_input=c["input"],
            system_prompt=c["system_prompt"],
            output_a=c["output_b"],
            output_b=c["output_a"],
            expected_output=c.get("expected_output", ""),
            cfg=cfg,
            swap_positions=False,
            inject_position_bias=True,
        )
        mapped_ba = {"A": "B", "B": "A", "TIE": "TIE"}.get(v_raw_ba.winner_ab, "TIE")
        if v_raw_ab.winner_ab != mapped_ba:
            unmitigated_flips += 1

        v_mit = judge_pair(
            case_id=c["id"],
            case_input=c["input"],
            system_prompt=c["system_prompt"],
            output_a=c["output_a"],
            output_b=c["output_b"],
            expected_output=c.get("expected_output", ""),
            cfg=cfg,
            swap_positions=True,
            inject_position_bias=False,
        )
        mitigated_verdicts.append(v_mit)

    # 2. Test-retest consistency (re-run suite at judge_temperature=0.0)
    retest_flips = 0
    for c, prev in zip(cases, mitigated_verdicts):
        v_re = judge_pair(
            case_id=c["id"],
            case_input=c["input"],
            system_prompt=c["system_prompt"],
            output_a=c["output_a"],
            output_b=c["output_b"],
            expected_output=c.get("expected_output", ""),
            cfg=cfg,
            swap_positions=True,
        )
        if v_re.final_winner != prev.final_winner:
            retest_flips += 1

    # 3. Adversarial probes (verbose-but-wrong, terse-but-correct, confidently-wrong, padded-answer)
    probe_results: List[Dict[str, Any]] = []
    unmitigated_verbosity_fooled = 0
    mitigated_verbosity_fooled = 0
    for p in probes:
        v_unmit = judge_pair(
            case_id=p["id"],
            case_input=p["input"],
            system_prompt=p["system_prompt"],
            output_a=p["output_a"],
            output_b=p["output_b"],
            expected_output=p.get("expected_output", ""),
            cfg=cfg,
            swap_positions=True,
            penalize_verbosity=False,
        )
        if v_unmit.final_winner != p["expected_winner"]:
            unmitigated_verbosity_fooled += 1

        v_mit_p = judge_pair(
            case_id=p["id"],
            case_input=p["input"],
            system_prompt=p["system_prompt"],
            output_a=p["output_a"],
            output_b=p["output_b"],
            expected_output=p.get("expected_output", ""),
            cfg=cfg,
            swap_positions=True,
            penalize_verbosity=True,
        )
        fooled = v_mit_p.final_winner != p["expected_winner"]
        if fooled:
            mitigated_verbosity_fooled += 1
        probe_results.append(
            {
                "id": p["id"],
                "probe_type": p["probe_type"],
                "expected_winner": p["expected_winner"],
                "judge_winner": v_mit_p.final_winner,
                "fooled": fooled,
                "score_a": v_mit_p.weighted_score_a,
                "score_b": v_mit_p.weighted_score_b,
                "rationale": v_mit_p.rationale,
            }
        )

    # 4. Aggregate A/B & Validation metrics
    gold_winners = [c["gold_winner"] for c in cases]
    pred_winners = [v.final_winner for v in mitigated_verdicts]
    agreements = sum(1 for g, p in zip(gold_winners, pred_winners) if g == p)
    kappa = _cohens_kappa(gold_winners, pred_winners)

    wins_a = sum(1 for v in mitigated_verdicts if v.final_winner == "A")
    wins_b = sum(1 for v in mitigated_verdicts if v.final_winner == "B")
    ties = sum(1 for v in mitigated_verdicts if v.final_winner == "TIE")

    mean_a = round(float(np.mean([v.weighted_score_a for v in mitigated_verdicts])), 3)
    mean_b = round(float(np.mean([v.weighted_score_b for v in mitigated_verdicts])), 3)
    pass_a = round(sum(1 for v in mitigated_verdicts if v.weighted_score_a >= 3.5) / len(cases), 4)
    pass_b = round(sum(1 for v in mitigated_verdicts if v.weighted_score_b >= 3.5) / len(cases), 4)

    std_scores = round(
        float(np.std([v.weighted_score_a for v in mitigated_verdicts] + [v.weighted_score_b for v in mitigated_verdicts])),
        3,
    )

    total_p_tok = sum(v.prompt_tokens for v in mitigated_verdicts)
    total_c_tok = sum(v.completion_tokens for v in mitigated_verdicts)

    report = {
        "config_a_name": suite.get("config_a_name", "prompt_v1_concise"),
        "config_b_name": suite.get("config_b_name", "prompt_v2_grounded_cot"),
        "judge_model": f"{cfg.judge_provider}:{cfg.judge_model}",
        "generator_model": f"{cfg.generator_provider}:{cfg.generator_model}",
        "num_cases": len(cases),
        "ab_comparison": {
            "config_a": {
                "pass_rate": pass_a,
                "mean_score": mean_a,
                "win_rate": round(wins_a / len(cases), 4),
                "wins": wins_a,
            },
            "config_b": {
                "pass_rate": pass_b,
                "mean_score": mean_b,
                "win_rate": round(wins_b / len(cases), 4),
                "wins": wins_b,
            },
            "ties": ties,
            "declared_winner": suite.get("config_b_name", "prompt_v2_grounded_cot")
            if wins_b > wins_a
            else suite.get("config_a_name", "prompt_v1_concise"),
        },
        "bias_metrics": {
            "position_flip_rate_before": round(unmitigated_flips / len(cases), 4),
            "position_flip_rate_after": round(
                sum(1 for v in mitigated_verdicts if v.position_flipped) / len(cases), 4
            ),
            "position_consensus_unresolved_ties": sum(
                1 for v in mitigated_verdicts if v.position_flipped
            ),
            "verbosity_probe_fooled_before": f"{unmitigated_verbosity_fooled}/{len(probes)}",
            "verbosity_probe_fooled_after": f"{mitigated_verbosity_fooled}/{len(probes)}",
            "score_spread_std": std_scores,
        },
        "validation": {
            "human_gold_agreement_rate": round(agreements / len(cases), 4),
            "cohens_kappa": kappa,
            "test_retest_flip_rate": round(retest_flips / len(cases), 4),
            "adversarial_probes": probe_results,
        },
        "cost_tracking": {
            "judge_calls_mitigated": len(cases) * 2,
            "prompt_tokens": total_p_tok,
            "completion_tokens": total_c_tok,
        },
        "verdicts": [asdict(v) for v in mitigated_verdicts],
    }

    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run suite and A/B comparison")
    p_run.add_argument("--suite", required=True, help="Path to suite.yaml or suite.json")
    p_run.add_argument("--out", default="judge_report.json", help="Output report JSON")

    args = parser.parse_args()
    if args.cmd == "run":
        report = run_evaluation_suite(args.suite, out_path=args.out)
        summary = {k: v for k, v in report.items() if k != "verdicts"}
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
