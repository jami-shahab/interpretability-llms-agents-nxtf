"""Error taxonomy — classifies failure modes found in MEPs.

Provides structured failure labels for post-hoc analysis:
  - planner_naics_mismatch
  - governance_missed_flag
  - benchmark_hallucinated_number
  - aggregator_rule_violation
  - parse_failure (per stage)

Usage:
    uv run ... -m sme_capital_eval.eval.error_taxonomy \
        --mep_dir meps/gemini_gemini_2_0_flash_lite/test \
        --out output/error_taxonomy.jsonl
"""

import argparse
import json
import os
import re
from typing import List

from ..mep.writer import iter_meps


# Sentinel words that suggest a hallucinated number (not from a tool)
_FABRICATION_PATTERNS = [
    r"industry average of \$?\d",
    r"typically \$?\d",
    r"benchmark of \$?\d",
    r"norm is \$?\d",
    r"standard is \d",
]


def _check_planner_naics(mep: dict) -> List[str]:
    """Flag if planner returned default fallback NAICS '00'."""
    plan = (mep.get("plan") or {}).get("parsed", {})
    errors = []
    if plan.get("enterprise_naics") == "00":
        errors.append("planner_naics_fallback_enterprise")
    if plan.get("project_naics") == "00":
        errors.append("planner_naics_fallback_project")
    return errors


def _check_governance_missed_flags(mep: dict) -> List[str]:
    """Flag if governance returned PASS but eval_metrics.must_flag is non-empty."""
    case = mep.get("case") or {}
    must_flag = case.get("eval_metrics", {}).get("must_flag", [])
    gov = (mep.get("governance") or {}).get("parsed", {})
    gov_flags = set(gov.get("flags", []))
    errors = []
    for f in must_flag:
        if f not in gov_flags:
            errors.append(f"governance_missed_flag:{f}")
    return errors


def _check_benchmark_hallucination(mep: dict) -> List[str]:
    """Flag if benchmark raw text contains suspicious hallucination patterns."""
    bm_raw = (mep.get("benchmark") or {}).get("raw_text", "").lower()
    errors = []
    for pattern in _FABRICATION_PATTERNS:
        if re.search(pattern, bm_raw, re.IGNORECASE):
            errors.append(f"benchmark_hallucinated_number:{pattern[:30]}")
    return errors


def _check_aggregator_rule_violation(mep: dict) -> List[str]:
    """Flag if LLM aggregator output disagrees with the deterministic rule engine."""
    agg = mep.get("aggregator") or {}
    rule_decision = agg.get("rule_decision", "")
    llm_decision = (agg.get("parsed") or {}).get("final_decision", "")
    if rule_decision and llm_decision and rule_decision != llm_decision:
        return [f"aggregator_rule_violation:rule={rule_decision},llm={llm_decision}"]
    return []


def _check_parse_failures(mep: dict) -> List[str]:
    """Flag parse errors per stage."""
    errors = []
    for stage in ["plan", "sponsor", "governance", "benchmark", "aggregator"]:
        obj = mep.get(stage) or {}
        if obj.get("parse_error"):
            errors.append(f"parse_failure:{stage}")
    return errors


def classify_mep(mep: dict) -> dict:
    """Run all error classifiers on one MEP and return a taxonomy report."""
    case = mep.get("case") or {}
    all_errors: List[str] = []
    all_errors += _check_planner_naics(mep)
    all_errors += _check_governance_missed_flags(mep)
    all_errors += _check_benchmark_hallucination(mep)
    all_errors += _check_aggregator_rule_violation(mep)
    all_errors += _check_parse_failures(mep)

    return {
        "case_id": case.get("case_id", ""),
        "expected_final": case.get("expected_final_decision", ""),
        "rule_decision": (mep.get("aggregator") or {}).get("rule_decision", ""),
        "error_count": len(all_errors),
        "errors": all_errors,
        "pipeline_errors": mep.get("errors", []),
    }


def main() -> None:
    """Classify errors in all MEPs and write taxonomy JSONL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mep_dir", required=True)
    parser.add_argument("--out", default="output/error_taxonomy.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total_errors = 0
    count = 0
    with open(args.out, "w") as f_out:
        for mep in iter_meps(args.mep_dir):
            result = classify_mep(mep)
            f_out.write(json.dumps(result) + "\n")
            total_errors += result["error_count"]
            count += 1

    print(f"Done. {count} MEPs classified, {total_errors} total error flags → {args.out}")


if __name__ == "__main__":
    main()
