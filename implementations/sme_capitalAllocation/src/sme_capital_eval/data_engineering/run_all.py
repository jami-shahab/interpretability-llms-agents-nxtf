"""Orchestrates the full data engineering pipeline.

Runs all ingestion steps in order:
  1. Policy documents  (fast — no network, extracted from docs/)
  2. Synthetic dataset (fast — local file split)
  3. FinQA             (network — HuggingFace download)
  4. StatsCan          (network — WDS API, ~2 table downloads)
  5. SFGSME            (network — WDS API, up to 39 table downloads)

Usage:
    uv run --env-file "$(git rev-parse --show-toplevel)/.env" \
        --directory "$(git rev-parse --show-toplevel)/implementations/sme_capitalAllocation" \
        -m sme_capital_eval.data_engineering.run_all \
        [--skip-sfgsme] [--skip-finqa]
"""

import argparse
import time

from .ingest_finqa import run as run_finqa
from .ingest_policy_docs import run as run_policy_docs
from .ingest_sfgsme import run as run_sfgsme
from .ingest_statscan import run as run_statscan
from .ingest_synthetic import run as run_synthetic


def main() -> None:
    """Run the full data engineering pipeline."""
    parser = argparse.ArgumentParser(description="SME Capital — full data engineering pipeline")
    parser.add_argument("--skip-sfgsme", action="store_true", help="Skip SFGSME (slow, 39 tables)")
    parser.add_argument("--skip-finqa", action="store_true", help="Skip FinQA (requires HuggingFace)")
    parser.add_argument("--skip-statscan", action="store_true", help="Skip StatsCan (requires network)")
    args = parser.parse_args()

    steps = [
        ("Policy Documents", run_policy_docs, False),
        ("Synthetic Dataset", run_synthetic, False),
        ("FinQA Few-Shot Bank", run_finqa, args.skip_finqa),
        ("StatsCan Capex + Ratios", run_statscan, args.skip_statscan),
        ("SFGSME Lending Data", run_sfgsme, args.skip_sfgsme),
    ]

    print("=" * 60)
    print("SME Capital Allocation — Data Engineering Pipeline")
    print("=" * 60)

    for name, fn, skip in steps:
        print(f"\n{'─'*60}")
        if skip:
            print(f"[SKIPPED] {name}")
            continue
        print(f"[RUNNING] {name}")
        t0 = time.perf_counter()
        try:
            fn()
            elapsed = time.perf_counter() - t0
            print(f"[OK] {name} completed in {elapsed:.1f}s")
        except Exception as exc:
            print(f"[ERROR] {name} failed: {exc}")

    print(f"\n{'='*60}")
    print("Data engineering pipeline complete.")
    print("Next step: run the MEP generation pipeline")
    print("  uv run -m sme_capital_eval.runner.run_generate_meps --split test --n 7")
    print("=" * 60)


if __name__ == "__main__":
    main()
