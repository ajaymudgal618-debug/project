# Problem 2 — LLM-as-Judge Evaluation Pipeline

A pairwise & rubric-based LLM-as-Judge evaluation pipeline with position-swap bias mitigation, verbosity/sycophancy probes, malformed-JSON recovery, and full audit logging.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Run suite + A/B comparison + bias & validation checks
python3 -m judge_pipeline.evaluate run --suite judge_pipeline/data/suite.yaml --out judge_report.json

# Run unit tests
python3 -m unittest discover -s tests -v
```

## Discussion
- **Before vs After Bias**: Running single-order evaluation with primacy bias produced measurable order flips; enabling bidirectional `(A, B)` + `(B, A)` consensus reduced unresolved position flips to 0% (disagreements resolve to `TIE`). Length-control penalties in the rubric prevented padded/verbose-but-wrong responses from beating terse-but-correct ones.
- **Release Gating**: Use the judge as a **pre-release regression signal** with automated guardrails (flip rate < 5%, Cohen's kappa >= 0.70, adversarial probe pass rate 100%, and JSON parse failure rate < 1%), keeping a **human-in-the-loop** for safety-critical changes, borderline `TIE` cases, and final release sign-off.
