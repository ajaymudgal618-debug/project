"""Core LLM-as-Judge engine with position-swap, length-control, and audit logging."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import JudgeConfig, get_judge_config
from .parser import CRITERIA_KEYS, ParsedVerdict, parse_verdict
from .rubric import RUBRIC, build_pairwise_prompt


@dataclass
class PairwiseVerdict:
    case_id: str
    winner_ab: str
    winner_ba_mapped: str
    final_winner: str  # A | B | TIE
    position_flipped: bool
    weighted_score_a: float
    weighted_score_b: float
    scores_a: Dict[str, float]
    scores_b: Dict[str, float]
    rationale: str
    prompt_tokens: int
    completion_tokens: int
    parse_statuses: List[str]


def _weighted_score(scores: Dict[str, float]) -> float:
    return round(sum(scores[k] * RUBRIC[k].weight for k in CRITERIA_KEYS), 3)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _score_single_local(
    output: str,
    expected: str,
    case_input: str,
    system_prompt: str,
    penalize_verbosity: bool = True,
) -> Dict[str, float]:
    """Calibrated local heuristic judge for offline/reproducible evaluation."""
    out_toks = _tokens(output)
    exp_toks = _tokens(expected)
    in_toks = _tokens(case_input)

    # Correctness: key-term recall + precision against expected_output
    if exp_toks:
        exp_set = set(exp_toks)
        out_set = set(out_toks)
        recall = len(exp_set & out_set) / len(exp_set)
        # Penalize contradictions / wrong numbers
        exp_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", expected))
        out_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", output))
        num_mismatch = len(exp_nums - out_nums) > 0 and len(out_nums - exp_nums) > 0
        correctness = 1.0 + 4.0 * recall - (1.8 if num_mismatch else 0.0)
    else:
        overlap = len(set(out_toks) & set(in_toks)) / max(1, len(set(in_toks)))
        correctness = 2.5 + 2.5 * min(1.0, overlap)

    # Faithfulness: penalize unsupported filler/padding relative to expected/input
    ref_set = set(exp_toks) | set(in_toks)
    supported_ratio = len(set(out_toks) & ref_set) / max(1, len(set(out_toks)))
    faithfulness = 1.5 + 3.5 * min(1.0, supported_ratio * 1.6)

    # Completeness
    completeness = 1.0 + 4.0 * (
        len(set(exp_toks) & set(out_toks)) / max(1, len(set(exp_toks)))
        if exp_toks
        else min(1.0, len(out_toks) / 25.0)
    )

    # Instruction following & verbosity penalty
    instr = 4.5
    if "concise" in system_prompt.lower() or "short" in system_prompt.lower():
        if len(out_toks) > 60:
            instr -= 1.5
    if penalize_verbosity and exp_toks and len(out_toks) > max(45, len(exp_toks) * 2.5):
        # Penalize padded/verbose answers
        instr -= 1.2
        faithfulness -= 0.8

    # Tone & safety: penalize sycophantic/confident-wrong markers
    sycophancy_markers = ("absolutely!", "great question!", "you are 100% right", "guaranteed")
    tone = 4.8
    if any(m in output.lower() for m in sycophancy_markers):
        tone -= 1.5

    return {
        "correctness": round(max(1.0, min(5.0, correctness)), 2),
        "faithfulness": round(max(1.0, min(5.0, faithfulness)), 2),
        "completeness": round(max(1.0, min(5.0, completeness)), 2),
        "instruction_following": round(max(1.0, min(5.0, instr)), 2),
        "tone_safety": round(max(1.0, min(5.0, tone)), 2),
    }


def _call_local_calibrated(
    prompt: str,
    case_input: str,
    system_prompt: str,
    output_a: str,
    output_b: str,
    expected_output: str,
    inject_position_bias: bool = False,
    penalize_verbosity: bool = True,
) -> tuple[str, int, int]:
    sa = _score_single_local(output_a, expected_output, case_input, system_prompt, penalize_verbosity)
    sb = _score_single_local(output_b, expected_output, case_input, system_prompt, penalize_verbosity)

    wa = _weighted_score(sa)
    wb = _weighted_score(sb)
    if inject_position_bias:
        # Unmitigated judges exhibit a ~0.25-point primacy bias toward Response A
        wa += 0.25

    if abs(wa - wb) < 0.15:
        winner = "TIE"
    elif wa > wb:
        winner = "A"
    else:
        winner = "B"

    rationale = (
        f"Weighted rubric score: A={wa:.2f}, B={wb:.2f}. "
        f"Correctness A={sa['correctness']}/B={sb['correctness']}, "
        f"Faithfulness A={sa['faithfulness']}/B={sb['faithfulness']}."
    )
    raw = json.dumps(
        {
            "scores_a": {k: round(v) for k, v in sa.items()},
            "scores_b": {k: round(v) for k, v in sb.items()},
            "winner": winner,
            "rationale": rationale,
        }
    )
    p_toks = len(prompt.split())
    c_toks = len(raw.split())
    return raw, p_toks, c_toks


def _call_remote_judge(prompt: str, cfg: JudgeConfig) -> tuple[str, int, int]:
    prov = cfg.judge_provider.lower()
    if prov == "openai":
        if not cfg.judge_api_key:
            raise ValueError("JUDGE_API_KEY required for OpenAI judge")
        payload = {
            "model": cfg.judge_model,
            "temperature": cfg.judge_temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.judge_api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = data["choices"][0]["message"]["content"]
        u = data.get("usage", {})
        return raw, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))

    if prov == "anthropic":
        if not cfg.judge_api_key:
            raise ValueError("JUDGE_API_KEY required for Anthropic judge")
        payload = {
            "model": cfg.judge_model,
            "max_tokens": 700,
            "temperature": cfg.judge_temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": cfg.judge_api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = data["content"][0]["text"]
        u = data.get("usage", {})
        return raw, int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))

    if prov == "gemini":
        if not cfg.judge_api_key:
            raise ValueError("JUDGE_API_KEY required for Gemini judge")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.judge_model}:generateContent?key={cfg.judge_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": cfg.judge_temperature},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        m = data.get("usageMetadata", {})
        return raw, int(m.get("promptTokenCount", 0)), int(m.get("candidatesTokenCount", 0))

    raise ValueError(f"Unsupported judge_provider: {cfg.judge_provider}")


def _log_audit(cfg: JudgeConfig, entry: Dict[str, Any]) -> None:
    parent = os.path.dirname(cfg.audit_log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(cfg.audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def judge_pair(
    case_id: str,
    case_input: str,
    system_prompt: str,
    output_a: str,
    output_b: str,
    expected_output: str = "",
    extra_criteria: str = "",
    cfg: Optional[JudgeConfig] = None,
    swap_positions: bool = True,
    inject_position_bias: bool = False,
    penalize_verbosity: bool = True,
) -> PairwiseVerdict:
    cfg = cfg or get_judge_config()

    def _invoke(oa: str, ob: str, order_tag: str) -> tuple[ParsedVerdict, int, int]:
        prompt = build_pairwise_prompt(
            case_input=case_input,
            system_prompt=system_prompt,
            output_a=oa,
            output_b=ob,
            expected_output=expected_output,
            extra_criteria=extra_criteria,
        )
        t0 = time.perf_counter()
        if cfg.judge_provider.lower() == "local_calibrated":
            raw, pt, ct = _call_local_calibrated(
                prompt,
                case_input,
                system_prompt,
                oa,
                ob,
                expected_output,
                inject_position_bias=inject_position_bias,
                penalize_verbosity=penalize_verbosity,
            )
        else:
            raw, pt, ct = _call_remote_judge(prompt, cfg)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        parsed = parse_verdict(raw)
        _log_audit(
            cfg,
            {
                "timestamp": time.time(),
                "case_id": case_id,
                "order": order_tag,
                "judge_provider": cfg.judge_provider,
                "judge_model": cfg.judge_model,
                "prompt": prompt,
                "raw_response": raw,
                "parse_status": parsed.parse_status,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "latency_ms": latency_ms,
            },
        )
        return parsed, pt, ct

    v_ab, pt1, ct1 = _invoke(output_a, output_b, "AB")

    if not swap_positions:
        wa = _weighted_score(v_ab.scores_a)
        wb = _weighted_score(v_ab.scores_b)
        return PairwiseVerdict(
            case_id=case_id,
            winner_ab=v_ab.winner,
            winner_ba_mapped=v_ab.winner,
            final_winner=v_ab.winner,
            position_flipped=False,
            weighted_score_a=wa,
            weighted_score_b=wb,
            scores_a=v_ab.scores_a,
            scores_b=v_ab.scores_b,
            rationale=v_ab.rationale,
            prompt_tokens=pt1,
            completion_tokens=ct1,
            parse_statuses=[v_ab.parse_status],
        )

    # Swapped order (B first, A second)
    v_ba, pt2, ct2 = _invoke(output_b, output_a, "BA")
    mapped_ba = {"A": "B", "B": "A", "TIE": "TIE"}.get(v_ba.winner, "TIE")
    flipped = v_ab.winner != mapped_ba

    avg_a = {k: round((v_ab.scores_a[k] + v_ba.scores_b[k]) / 2.0, 2) for k in CRITERIA_KEYS}
    avg_b = {k: round((v_ab.scores_b[k] + v_ba.scores_a[k]) / 2.0, 2) for k in CRITERIA_KEYS}
    wa = _weighted_score(avg_a)
    wb = _weighted_score(avg_b)

    if not flipped:
        final_winner = v_ab.winner
    else:
        # Require agreement; fall back to TIE when position-swapped orders disagree
        final_winner = "TIE"

    return PairwiseVerdict(
        case_id=case_id,
        winner_ab=v_ab.winner,
        winner_ba_mapped=mapped_ba,
        final_winner=final_winner,
        position_flipped=flipped,
        weighted_score_a=wa,
        weighted_score_b=wb,
        scores_a=avg_a,
        scores_b=avg_b,
        rationale=f"[AB:{v_ab.winner} | BA->orig:{mapped_ba}] {v_ab.rationale}",
        prompt_tokens=pt1 + pt2,
        completion_tokens=ct1 + ct2,
        parse_statuses=[v_ab.parse_status, v_ba.parse_status],
    )
