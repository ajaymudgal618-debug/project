"""Independent configuration for Judge and Generator models."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeConfig:
    judge_provider: str = os.getenv("JUDGE_PROVIDER", "local_calibrated")
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-3-5-sonnet-20241022")
    judge_api_key: str = os.getenv("JUDGE_API_KEY", "")
    judge_temperature: float = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))

    generator_provider: str = os.getenv("GENERATOR_PROVIDER", "openai")
    generator_model: str = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")
    generator_api_key: str = os.getenv("GENERATOR_API_KEY", "")

    audit_log_path: str = os.getenv("JUDGE_AUDIT_LOG", "logs/judge_audit.jsonl")


def get_judge_config() -> JudgeConfig:
    return JudgeConfig(
        judge_provider=os.getenv("JUDGE_PROVIDER", "local_calibrated"),
        judge_model=os.getenv("JUDGE_MODEL", "claude-3-5-sonnet-20241022"),
        judge_api_key=os.getenv("JUDGE_API_KEY", ""),
        judge_temperature=float(os.getenv("JUDGE_TEMPERATURE", "0.0")),
        generator_provider=os.getenv("GENERATOR_PROVIDER", "openai"),
        generator_model=os.getenv("GENERATOR_MODEL", "gpt-4o-mini"),
        generator_api_key=os.getenv("GENERATOR_API_KEY", ""),
        audit_log_path=os.getenv("JUDGE_AUDIT_LOG", "logs/judge_audit.jsonl"),
    )
