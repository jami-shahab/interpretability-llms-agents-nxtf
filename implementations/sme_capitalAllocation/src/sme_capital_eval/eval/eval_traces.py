"""Pass 2: trace-based evaluation — latency, parse reliability, pipeline health.

Usage:
    uv run ... -m sme_capital_eval.eval.eval_traces \
        --mep_dir meps/gemini_gemini_2_0_flash_lite/test \
        --out output/trace_metrics.jsonl
"""

import argparse
import json
import os

from ..mep.writer import iter_meps


def evaluate_trace(mep: dict) -> dict:
    """Extract trace-level metrics from a single MEP.

    Parameters
    ----------
    mep : dict
        Parsed MEP dict.

    Returns
    -------
    dict
        Flat dict of latency + reliability metrics.
    """
    ts = mep.get("timestamps", {}) or {}
    case = mep.get("case", {}) or {}

    stage_ms = {
        "planner_ms": ts.get("planner_ms", 0) or 0,
        "sponsor_ms": ts.get("sponsor_ms", 0) or 0,
        "governance_ms": ts.get("governance_ms", 0) or 0,
        "benchmark_ms": ts.get("benchmark_ms", 0) or 0,
        "aggregator_ms": ts.get("aggregator_ms", 0) or 0,
    }
    total_ms = sum(stage_ms.values())

    parse_flags = {
        "planner_parse_ok": not (mep.get("plan") or {}).get("parse_error", True),
        "sponsor_parse_ok": not (mep.get("sponsor") or {}).get("parse_error", True),
        "governance_parse_ok": not (mep.get("governance") or {}).get("parse_error", True),
        "benchmark_parse_ok": not (mep.get("benchmark") or {}).get("parse_error", True),
        "aggregator_parse_ok": not (mep.get("aggregator") or {}).get("parse_error", True),
    }
    parse_reliability = sum(parse_flags.values()) / len(parse_flags)

    errors = mep.get("errors", [])

    return {
        "case_id": case.get("case_id", ""),
        "total_latency_sec": round(total_ms / 1000.0, 3),
        **stage_ms,
        **parse_flags,
        "parse_reliability": round(parse_reliability, 4),
        "error_count": len(errors),
        "errors": errors,
        "lf_trace_id": mep.get("lf_trace_id"),
    }


def main() -> None:
    """Evaluate MEP traces and write to JSONL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mep_dir", required=True)
    parser.add_argument("--out", default="output/trace_metrics.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    count = 0
    with open(args.out, "w") as f_out:
        for mep in iter_meps(args.mep_dir):
            try:
                m = evaluate_trace(mep)
                f_out.write(json.dumps(m) + "\n")
                count += 1
            except Exception as exc:
                print(f"  Error: {exc}")

    print(f"Done. {count} trace metrics written to {args.out}")


if __name__ == "__main__":
    main()
