"""Pass 1: output-based evaluation — decision accuracy + per-lens + must-flag/cite.

Usage:
    uv run --env-file "$(git rev-parse --show-toplevel)/.env" \
        --directory "$(git rev-parse --show-toplevel)/implementations/sme_capitalAllocation" \
        -m sme_capital_eval.eval.eval_outputs \
        --mep_dir meps/gemini_gemini_2_0_flash_lite/test \
        --out output/metrics.jsonl
"""

import argparse
import json
import os
from typing import Optional

from dotenv import load_dotenv

from ..mep.writer import iter_meps


load_dotenv()


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def score_final_decision(expected: str, predicted: str) -> float:
    """Exact-match accuracy for the 4-class final decision.

    Returns 1.0 on match, 0.0 otherwise.
    """
    return 1.0 if expected.strip().upper() == predicted.strip().upper() else 0.0


def score_per_lens(
    expected_per_lens: dict,
    sponsor_parsed: dict,
    governance_parsed: dict,
    benchmark_parsed: dict,
) -> dict:
    """Compute per-agent accuracy vs ground truth per_lens_expected.

    Returns
    -------
    dict
        {sponsor_correct: float, governance_correct: float, benchmark_correct: float,
         per_lens_accuracy: float}
    """
    results = {}
    agent_map = {
        "sponsor": sponsor_parsed,
        "governance": governance_parsed,
        "benchmark": benchmark_parsed,
    }
    correct = 0
    total = 0
    for lens, parsed in agent_map.items():
        exp = expected_per_lens.get(lens, "").strip().upper()
        pred = parsed.get("decision", "").strip().upper()
        match = 1.0 if exp == pred else 0.0
        results[f"{lens}_correct"] = match
        if exp:
            correct += match
            total += 1
    results["per_lens_accuracy"] = round(correct / total, 4) if total > 0 else 0.0
    return results


# (must_flag and must_cite scorers removed in favor of direct metric counting)


# ---------------------------------------------------------------------------
# Per-MEP evaluation
# ---------------------------------------------------------------------------


def evaluate_mep(mep: dict, use_judge: bool = False) -> dict:
    """Evaluate a single MEP and return a flat metrics dict."""
    case = mep.get("case", {})
    plan = mep.get("plan", {})
    sponsor = mep.get("sponsor", {}) or {}
    governance = mep.get("governance", {}) or {}
    benchmark = mep.get("benchmark", {}) or {}
    aggregator = mep.get("aggregator", {}) or {}
    timestamps = mep.get("timestamps", {}) or {}
    config = mep.get("config", {}) or {}

    expected_final = case.get("expected_final_decision", "")
    expected_per_lens = case.get("expected_per_lens", {})
    eval_metrics = case.get("eval_metrics", {})

    sp_parsed = sponsor.get("parsed", {}) or {}
    gov_parsed = governance.get("parsed", {}) or {}
    bm_parsed = benchmark.get("parsed", {}) or {}
    agg_parsed = aggregator.get("parsed", {}) or {}

    predicted_final = agg_parsed.get("final_decision", aggregator.get("rule_decision", ""))

    # Timing
    total_ms = sum([
        timestamps.get("planner_ms", 0) or 0,
        timestamps.get("sponsor_ms", 0) or 0,
        timestamps.get("governance_ms", 0) or 0,
        timestamps.get("benchmark_ms", 0) or 0,
        timestamps.get("aggregator_ms", 0) or 0,
    ])

    per_lens = score_per_lens(expected_per_lens, sp_parsed, gov_parsed, bm_parsed)

    metrics: dict = {
        "case_id": case.get("case_id", ""),
        "industry": case.get("industry", ""),
        "asset_archetype": case.get("asset_archetype", ""),
        "config_name": config.get("config_name", ""),
        "expected_final": expected_final,
        "predicted_final": predicted_final,
        "rule_decision": aggregator.get("rule_decision", ""),
        "rule_applied": aggregator.get("rule_applied", ""),
        "final_decision_correct": score_final_decision(expected_final, predicted_final),
        **per_lens,
        "governance_flags_raised": len(gov_parsed.get("flags", [])),
        "benchmark_tool_calls": len(benchmark.get("tool_trace", [])),
        "override_applied": agg_parsed.get("override_applied", False),
        "override_direction": agg_parsed.get("override_direction", ""),
        "latency_sec": round(total_ms / 1000.0, 3),
        "planner_ms": timestamps.get("planner_ms", 0),
        "sponsor_ms": timestamps.get("sponsor_ms", 0),
        "governance_ms": timestamps.get("governance_ms", 0),
        "benchmark_ms": timestamps.get("benchmark_ms", 0),
        "aggregator_ms": timestamps.get("aggregator_ms", 0),
        "planner_parse_ok": not plan.get("parse_error", True),
        "sponsor_parse_ok": not sponsor.get("parse_error", True),
        "governance_parse_ok": not governance.get("parse_error", True),
        "benchmark_parse_ok": not benchmark.get("parse_error", True),
        "aggregator_parse_ok": not aggregator.get("parse_error", True),
        "all_parse_ok": not any([
            plan.get("parse_error", True),
            sponsor.get("parse_error", True),
            governance.get("parse_error", True),
            benchmark.get("parse_error", True),
            aggregator.get("parse_error", True),
        ]),
        "has_errors": bool(mep.get("errors", [])),
        "error_count": len(mep.get("errors", [])),
    }

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Evaluate MEPs and write metrics to JSONL."""
    parser = argparse.ArgumentParser(description="Evaluate SME Capital MEPs — output metrics")
    parser.add_argument("--mep_dir", required=True)
    parser.add_argument("--out", default="output/metrics.jsonl")
    parser.add_argument("--no_judge", action="store_true", help="Skip LLM judge (default: skip)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    count = 0
    with open(args.out, "w") as f_out:
        for mep in iter_meps(args.mep_dir):
            try:
                m = evaluate_mep(mep, use_judge=not args.no_judge)
                f_out.write(json.dumps(m) + "\n")
                count += 1
            except Exception as exc:
                print(f"  Error evaluating MEP: {exc}")

    print(f"Done. {count} metrics written to {args.out}")


if __name__ == "__main__":
    main()
