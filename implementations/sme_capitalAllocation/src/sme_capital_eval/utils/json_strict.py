"""Strict JSON parser with json_repair fallback."""

import json
from typing import Any, Optional, Tuple


def parse_strict(
    text: str,
    required_keys: Optional[list] = None,
) -> Tuple[dict, bool]:
    """Parse JSON from LLM output; fall back to json_repair if needed.

    Parameters
    ----------
    text : str
        Raw LLM output text.
    required_keys : list, optional
        Keys that must be present for parse to be considered successful.

    Returns
    -------
    parsed : dict
        Parsed result, or empty dict on total failure.
    ok : bool
        True if parsing succeeded and all required keys are present.
    """
    # Strip markdown fences
    cleaned = text.strip()
    for fence in ("```json", "```JSON", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
            break
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try strict JSON first
    parsed: Any = {}
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json  # type: ignore
            parsed = json.loads(repair_json(cleaned))
        except Exception:
            return {}, False

    if not isinstance(parsed, dict):
        return {}, False

    if required_keys:
        missing = [k for k in required_keys if k not in parsed]
        if missing:
            return parsed, False

    return parsed, True
