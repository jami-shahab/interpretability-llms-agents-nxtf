"""Ingest and split the NXTFrontier synthetic dataset.

Reads ``docs/synthetic_dataset.json``, validates the schema,
splits into fewshot / test sets, and writes processed copies to
``data/processed/``.

Usage:
    uv run ... -m sme_capital_eval.data_engineering.ingest_synthetic
"""

import json
from pathlib import Path


# parents[3] = implementations/sme_capitalAllocation/
_ROOT = Path(__file__).parents[3]
_SOURCE = _ROOT / "docs" / "synthetic_dataset.json"
_OUT_DIR = _ROOT / "data" / "processed"

FEW_SHOT_IDS = {"CASE_01", "CASE_04", "CASE_07"}
TEST_IDS = {"CASE_02", "CASE_03", "CASE_05", "CASE_06", "CASE_08", "CASE_09", "CASE_10"}

REQUIRED_CASE_KEYS = {
    "case_id",
    "company_profile",
    "expansion_plan",
    "financials",
    "expected_outcome",
    "eval_metrics",
}


def _validate(cases: list) -> list:
    """Validate schema and return list of issues."""
    issues = []
    for c in cases:
        missing = REQUIRED_CASE_KEYS - set(c.keys())
        if missing:
            issues.append(f"{c.get('case_id', '?')}: missing keys {missing}")
        if "final_decision" not in c.get("expected_outcome", {}):
            issues.append(f"{c.get('case_id', '?')}: expected_outcome.final_decision missing")
    return issues


def run() -> None:
    """Execute ingestion: validate → split → write processed files."""
    print(f"Source : {_SOURCE}")
    if not _SOURCE.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {_SOURCE}")

    with open(_SOURCE) as f:
        cases = json.load(f)

    print(f"Loaded : {len(cases)} cases")

    issues = _validate(cases)
    if issues:
        print("  WARNINGS:")
        for i in issues:
            print(f"    - {i}")
    else:
        print("  Schema validation: OK")

    fewshot = [c for c in cases if c["case_id"] in FEW_SHOT_IDS]
    test = [c for c in cases if c["case_id"] in TEST_IDS]

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, subset in [("fewshot_cases.json", fewshot), ("test_cases.json", test)]:
        path = _OUT_DIR / name
        with open(path, "w") as f:
            json.dump(subset, f, indent=2)
        decisions = [c["expected_outcome"]["final_decision"] for c in subset]
        print(f"  → {path.name} : {len(subset)} cases | {decisions}")


if __name__ == "__main__":
    run()
