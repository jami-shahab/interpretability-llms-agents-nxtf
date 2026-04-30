"""Loader for NXTFrontier synthetic capital allocation cases.

Reads ``synthetic_dataset.json`` (or the processed split files in ``data/processed/``)
and returns lists of :class:`CapitalCase` objects ready for the pipeline.
"""

import json
from pathlib import Path
from typing import List, Optional

from .capital_case import CapitalCase


# IDs reserved as in-context few-shot examples (one per decision class)
FEW_SHOT_IDS = {"CASE_01", "CASE_04", "CASE_07"}

# Remaining cases form the evaluation test set
TEST_IDS = {"CASE_02", "CASE_03", "CASE_05", "CASE_06", "CASE_08", "CASE_09", "CASE_10"}

# parents[3] = implementations/sme_capitalAllocation/
_SME_ROOT = Path(__file__).parents[3]
_SYNTHETIC_PATH = _SME_ROOT / "docs" / "synthetic_dataset.json"
_PROCESSED_DIR = _SME_ROOT / "data" / "processed"


def _raw_to_case(raw: dict) -> CapitalCase:
    """Convert a raw JSON dict from the synthetic dataset to a :class:`CapitalCase`."""
    return CapitalCase(
        case_id=raw["case_id"],
        company_profile=raw.get("company_profile", {}),
        expansion_plan=raw.get("expansion_plan", {}),
        financials=raw.get("financials", {}),
        financing_details=raw.get("financing_details", {}),
        approval_context=raw.get("approval_context", {}),
        constraint_formalization=raw.get("constraint_formalization", []),
        external_context=raw.get("external_context", {}),
        planner_answerability=raw.get("planner_answerability", {}),
        expected_outcome=raw.get("expected_outcome", {}),
        eval_metrics=raw.get("eval_metrics", {}),
        metadata=raw.get("metadata", {}),
    )


def _load_all_raw(source_path: Optional[Path] = None) -> List[dict]:
    """Load all raw case dicts from the synthetic dataset JSON."""
    path = source_path or _SYNTHETIC_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found at {path}. "
            "Run the data engineering pipeline first: "
            "uv run -m sme_capital_eval.data_engineering.run_all"
        )
    with open(path) as f:
        return json.load(f)


def load_cases(
    split: str = "test",
    source_path: Optional[Path] = None,
    n: Optional[int] = None,
) -> List[CapitalCase]:
    """Load cases for the given split.

    Parameters
    ----------
    split : {'test', 'fewshot', 'all'}
        Which subset to return.
    source_path : Path, optional
        Override the default synthetic dataset path.
    n : int, optional
        Truncate to first N cases.

    Returns
    -------
    List[CapitalCase]
        Ordered list of cases for the requested split.
    """
    raw_cases = _load_all_raw(source_path)

    if split == "fewshot":
        selected = [r for r in raw_cases if r["case_id"] in FEW_SHOT_IDS]
    elif split == "test":
        selected = [r for r in raw_cases if r["case_id"] in TEST_IDS]
    else:  # "all"
        selected = raw_cases

    cases = [_raw_to_case(r) for r in selected]
    if n is not None:
        cases = cases[:n]
    return cases


def load_fewshot_cases(source_path: Optional[Path] = None) -> List[CapitalCase]:
    """Convenience wrapper — returns the 3 few-shot cases (PROCEED, DECLINE, PROCEED_WITH_MITIGATION)."""
    return load_cases(split="fewshot", source_path=source_path)
