"""Aggregate metrics.jsonl → summary.csv.

Usage:
    uv run ... -m sme_capital_eval.eval.summarize \
        --metrics output/metrics.jsonl \
        --out output/summary.csv
"""

import argparse
import json

import pandas as pd


def main() -> None:
    """Summarise metrics.jsonl into a human-readable CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="output/metrics.jsonl")
    parser.add_argument("--out", default="output/summary.csv")
    args = parser.parse_args()

    rows = []
    with open(args.metrics) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("No rows found in metrics file.")
        return

    df = pd.DataFrame(rows)

    numeric_cols = [
        "final_decision_correct",
        "per_lens_accuracy",
        "sponsor_correct",
        "governance_correct",
        "benchmark_correct",
        "must_flag_hit_rate",
        "must_cite_hit_rate",
        "latency_sec",
        "all_parse_ok",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    summary = df[numeric_cols].agg(["mean", "min", "max", "std"]).round(4)
    summary.to_csv(args.out)

    # Print readable summary
    n = len(df)
    print(f"\n{'='*50}")
    print(f"SME Capital Eval Summary — {n} cases")
    print(f"{'='*50}")
    if "final_decision_correct" in df.columns:
        acc = df["final_decision_correct"].mean()
        print(f"Final Decision Accuracy : {acc:.1%}  ({int(acc*n)}/{n})")
    if "per_lens_accuracy" in df.columns:
        print(f"Per-Lens Accuracy       : {df['per_lens_accuracy'].mean():.1%}")
    if "must_flag_hit_rate" in df.columns:
        print(f"Must-Flag Hit Rate      : {df['must_flag_hit_rate'].mean():.1%}")
    if "must_cite_hit_rate" in df.columns:
        print(f"Must-Cite Hit Rate      : {df['must_cite_hit_rate'].mean():.1%}")
    if "latency_sec" in df.columns:
        print(f"Avg Latency (s)         : {df['latency_sec'].mean():.2f}")
    if "all_parse_ok" in df.columns:
        print(f"Parse Reliability       : {df['all_parse_ok'].mean():.1%}")
    print(f"{'='*50}")
    print(f"\nPer-case breakdown:")
    display_cols = ["case_id", "expected_final", "predicted_final",
                    "final_decision_correct", "per_lens_accuracy", "latency_sec"]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))
    print(f"\nSummary CSV written to: {args.out}")


if __name__ == "__main__":
    main()
