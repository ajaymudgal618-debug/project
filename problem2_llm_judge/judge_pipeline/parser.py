"""Robust structured verdict parser with multi-stage malformed JSON recovery."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

CRITERIA_KEYS = (
    "correctness",
    "faithfulness",
    "completeness",
    "instruction_following",
    "tone_safety",
)


@dataclass
class ParsedVerdict:
    scores_a: Dict[str, float]
    scores_b: Dict[str, float]
    winner: str
    rationale: str
    parse_status: str  # exact | fence_stripped | brace_repaired | regex_fallback | failed


def _strip_fences(raw: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    return m.group(1).strip() if m else raw.strip()


def _extract_balanced_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
    return None


def _repair_json_str(s: str) -> str:
    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Replace smart quotes if any
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return s


def _normalize_scores(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in CRITERIA_KEYS:
        val = d.get(k, 3.0)
        try:
            num = float(val)
        except (TypeError, ValueError):
            num = 3.0
        out[k] = max(1.0, min(5.0, num))
    return out


def parse_verdict(raw_response: str) -> ParsedVerdict:
    text = raw_response.strip()

    # Stage 1: Direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "winner" in obj:
            return ParsedVerdict(
                scores_a=_normalize_scores(obj.get("scores_a", {})),
                scores_b=_normalize_scores(obj.get("scores_b", {})),
                winner=str(obj.get("winner", "TIE")).upper(),
                rationale=str(obj.get("rationale", "")),
                parse_status="exact",
            )
    except json.JSONDecodeError:
        pass

    # Stage 2: Strip markdown code fences
    unfenced = _strip_fences(text)
    try:
        obj = json.loads(unfenced)
        if isinstance(obj, dict) and "winner" in obj:
            return ParsedVerdict(
                scores_a=_normalize_scores(obj.get("scores_a", {})),
                scores_b=_normalize_scores(obj.get("scores_b", {})),
                winner=str(obj.get("winner", "TIE")).upper(),
                rationale=str(obj.get("rationale", "")),
                parse_status="fence_stripped",
            )
    except json.JSONDecodeError:
        pass

    # Stage 3: Extract balanced braces + repair trailing commas
    candidate = _extract_balanced_json(unfenced)
    if candidate:
        repaired = _repair_json_str(candidate)
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return ParsedVerdict(
                    scores_a=_normalize_scores(obj.get("scores_a", {})),
                    scores_b=_normalize_scores(obj.get("scores_b", {})),
                    winner=str(obj.get("winner", "TIE")).upper(),
                    rationale=str(obj.get("rationale", "")),
                    parse_status="brace_repaired",
                )
        except json.JSONDecodeError:
            pass

    # Stage 4: Regex field-level recovery
    w_match = re.search(r'"?winner"?\s*[:=]\s*"?([AB]|TIE)"?', text, re.IGNORECASE)
    if w_match:
        winner = w_match.group(1).upper()
        r_match = re.search(r'"?rationale"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
        rationale = r_match.group(1) if r_match else "Recovered via regex fallback."
        default_scores = {k: 3.0 for k in CRITERIA_KEYS}
        return ParsedVerdict(
            scores_a=default_scores,
            scores_b=dict(default_scores),
            winner=winner,
            rationale=rationale,
            parse_status="regex_fallback",
        )

    # Complete failure fallback
    default_scores = {k: 3.0 for k in CRITERIA_KEYS}
    return ParsedVerdict(
        scores_a=default_scores,
        scores_b=dict(default_scores),
        winner="TIE",
        rationale="Malformed judge output; defaulted to TIE.",
        parse_status="failed",
    )
