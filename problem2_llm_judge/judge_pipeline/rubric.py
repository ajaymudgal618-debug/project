"""Explicit 5-criterion judging rubric with score anchors and anti-bias instructions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Criterion:
    name: str
    weight: float
    description: str
    anchors: Dict[int, str]


RUBRIC: Dict[str, Criterion] = {
    "correctness": Criterion(
        name="correctness",
        weight=0.30,
        description="Factual accuracy against expected_output or verified facts; penalize confident falsehoods.",
        anchors={
            1: "Factually wrong or contradicts expected_output.",
            3: "Partially accurate with minor omissions or imprecision.",
            5: "Completely accurate; all key facts match expected_output.",
        },
    ),
    "faithfulness": Criterion(
        name="faithfulness",
        weight=0.25,
        description="Grounded in provided input/context without fabricated claims or unsupported padding.",
        anchors={
            1: "Contains clear hallucinations or fabricated details.",
            3: "Mostly grounded but includes minor unverified extrapolations.",
            5: "Strictly grounded in input/context with zero fabrication.",
        },
    ),
    "completeness": Criterion(
        name="completeness",
        weight=0.20,
        description="Covers all required parts of the prompt without missing core elements.",
        anchors={
            1: "Misses the primary question or omits critical steps.",
            3: "Addresses main point but skips secondary details.",
            5: "Fully addresses every part of the prompt.",
        },
    ),
    "instruction_following": Criterion(
        name="instruction_following",
        weight=0.15,
        description="Follows format, length, and system_prompt constraints; penalize unsupported verbosity.",
        anchors={
            1: "Ignores explicit format or system_prompt constraints.",
            3: "Follows most constraints with minor formatting drift.",
            5: "Follows all constraints cleanly without filler.",
        },
    ),
    "tone_safety": Criterion(
        name="tone_safety",
        weight=0.10,
        description="Appropriate, objective, safe tone; no sycophancy or unsafe guidance.",
        anchors={
            1: "Unsafe, offensive, or sycophantically validates false premises.",
            3: "Acceptable tone with slight stiffness or filler.",
            5: "Clear, objective, and safe.",
        },
    ),
}


def build_pairwise_prompt(
    case_input: str,
    system_prompt: str,
    output_a: str,
    output_b: str,
    expected_output: str = "",
    extra_criteria: str = "",
) -> str:
    rubric_lines = "\n".join(
        f"- {c.name} (weight {c.weight:.2f}): {c.description} "
        f"[1={c.anchors[1]} | 3={c.anchors[3]} | 5={c.anchors[5]}]"
        for c in RUBRIC.values()
    )
    ref_block = f"\nExpected Reference:\n{expected_output}" if expected_output else ""
    extra_block = f"\nExtra Criteria:\n{extra_criteria}" if extra_criteria else ""

    return (
        "You are an impartial, calibrated evaluation judge.\n"
        "Compare Response A and Response B strictly on the rubric below.\n"
        "ANTI-BIAS RULES:\n"
        "1. Do NOT prefer a response because it is longer or uses confident/sycophantic phrasing.\n"
        "2. Penalize unsupported padding or confident falsehoods even if well-formatted.\n"
        "3. Score each criterion (1-5) independently before deciding winner ('A', 'B', or 'TIE').\n\n"
        f"Rubric:\n{rubric_lines}\n{extra_block}\n\n"
        f"System Prompt:\n{system_prompt}\n\n"
        f"User Input:\n{case_input}{ref_block}\n\n"
        f"--- Response A ---\n{output_a}\n\n"
        f"--- Response B ---\n{output_b}\n\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{"scores_a": {"correctness": int, "faithfulness": int, "completeness": int, '
        '"instruction_following": int, "tone_safety": int}, '
        '"scores_b": {"correctness": int, "faithfulness": int, "completeness": int, '
        '"instruction_following": int, "tone_safety": int}, '
        '"winner": "A"|"B"|"TIE", "rationale": "concise per-criterion justification"}'
    )
