"""Ingest FinQA dataset from HuggingFace and curate a static few-shot bank.

Filters for samples that demonstrate leverage/ratio calculations, formats
them into 4-part few-shot blocks, and writes to:
  ``data/processed/finqa_fewshot_bank.jsonl``

These are used as math grounding examples in agent system prompts.
The selection is STATIC (curated once) — not dynamic retrieval.

Usage:
    uv run ... -m sme_capital_eval.data_engineering.ingest_finqa
"""

import json
import re
from pathlib import Path


# parents[3] = implementations/sme_capitalAllocation/
_ROOT = Path(__file__).parents[3]
_OUT_DIR = _ROOT / "data" / "processed"
_CACHE_DIR = _ROOT / "data" / "raw" / "finqa"

_OUT_FILE = _OUT_DIR / "finqa_fewshot_bank.jsonl"

# Keywords that indicate financial ratio / leverage calculations
RATIO_KEYWORDS = [
    "debt", "leverage", "ratio", "ebitda", "payback", "irr",
    "interest coverage", "current ratio", "equity", "return on",
    "capital", "cash flow", "operating income", "net income",
]

# Maximum examples to keep in the bank (keeps prompts lean)
MAX_EXAMPLES = 10


def _has_ratio_content(example: dict) -> bool:
    """Return True if the example involves ratio/leverage-type reasoning."""
    q = example.get("qa", {}).get("question", "").lower()
    pre = " ".join(example.get("pre_text", [])).lower()
    combined = q + " " + pre
    return any(kw in combined for kw in RATIO_KEYWORDS)


def _format_fewshot(example: dict, idx: int) -> dict:
    """Format one FinQA example as a compact 4-part few-shot block.

    Returns a dict with keys: id, question, steps, answer, formatted_block.
    """
    qa = example.get("qa", {})
    question = qa.get("question", "")
    steps = qa.get("steps", [])
    answer = str(qa.get("exe_ans", ""))

    # Format steps as numbered list
    step_text = "\n".join(
        f"  Step {i+1}: {s}" for i, s in enumerate(steps)
    ) if steps else "  (No explicit steps recorded)"

    # Build compact block
    block = (
        f"Q: {question}\n"
        f"Steps:\n{step_text}\n"
        f"Answer: {answer}"
    )

    return {
        "id": f"finqa_{idx:04d}",
        "question": question,
        "steps": steps,
        "answer": answer,
        "formatted_block": block,
    }


def run() -> None:
    """Download FinQA, filter for ratio examples, write few-shot bank."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FinQA from HuggingFace (ibm-research/finqa) …")
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("ibm-research/finqa", split="train", cache_dir=str(_CACHE_DIR))
    except Exception as exc:
        print(f"  ERROR: Could not load FinQA: {exc}")
        print("  Ensure 'datasets' is installed and network is available.")
        _write_empty_bank()
        return

    print(f"  Loaded {len(ds)} training examples")

    # Filter
    filtered = [ex for ex in ds if _has_ratio_content(ex)]
    print(f"  After ratio/leverage filter: {len(filtered)} examples")

    # Sort by step count (prefer examples with explicit calculation steps)
    filtered.sort(key=lambda x: len(x.get("qa", {}).get("steps", [])), reverse=True)

    # Take top MAX_EXAMPLES
    selected = filtered[:MAX_EXAMPLES]
    print(f"  Selected top {len(selected)} examples for few-shot bank")

    with open(_OUT_FILE, "w") as f:
        for i, ex in enumerate(selected):
            block = _format_fewshot(ex, i)
            f.write(json.dumps(block) + "\n")

    print(f"\nFinQA few-shot bank written to: {_OUT_FILE}")

    # Print preview of first 2
    print("\nPreview (first 2 examples):")
    with open(_OUT_FILE) as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            item = json.loads(line)
            print(f"\n[{item['id']}] {item['question']}")
            print(f"Answer: {item['answer']}")


def _write_empty_bank() -> None:
    """Write an empty bank file so downstream code doesn't fail."""
    _OUT_FILE.write_text("")
    print(f"  Empty bank file written to {_OUT_FILE}")


if __name__ == "__main__":
    run()
