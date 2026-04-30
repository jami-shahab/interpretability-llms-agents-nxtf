"""LLM-as-judge rubric scorer for capital allocation MEPs.

Evaluates 4 dimensions:
- rationale_specificity   : Does the rationale cite specific numbers/thresholds?
- constraint_grounding    : Does governance/benchmark cite actual policy clauses?
- decision_logic          : Is the final decision consistent with sub-agent outputs?
- no_fabrication          : Did agents avoid inventing benchmark numbers?

Uses gemini-2.0-flash-lite (cheapest model) to minimise quota consumption.
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv


load_dotenv()

JUDGE_PROMPT = """You are an audit-grade evaluator for a capital allocation AI system.

Score the following agent pipeline output on 4 dimensions (0.0 to 1.0 each):

1. rationale_specificity: Does the aggregator rationale cite specific numbers (e.g., leverage ratio, cash amount)? 1.0=all key numbers cited, 0.0=vague generalities only.

2. constraint_grounding: Does the governance agent mention the specific policy rule that was violated (e.g., "Treasury Policy Section 4", "2.5x ceiling", "SAMP Section 5")? 1.0=explicit policy reference, 0.5=partial, 0.0=none.

3. decision_logic: Is the final_decision logically consistent with the sub-agent decisions and flags? 1.0=fully consistent, 0.5=minor inconsistency, 0.0=contradicts agent outputs.

4. no_fabrication: Did the benchmark agent avoid making up benchmark numbers? 1.0=only retrieved data or explicit "data unavailable", 0.0=clearly invented numbers.

[AGENT OUTPUTS TO EVALUATE]
{outputs_json}

Output ONLY a JSON object with exactly these keys:
{{"rationale_specificity": <0-1>, "constraint_grounding": <0-1>, "decision_logic": <0-1>, "no_fabrication": <0-1>, "judge_notes": "<one sentence>"}}"""


def judge_mep(
    mep: dict,
    model: str = "gemini-2.0-flash-lite",
) -> dict:
    """Score a MEP on 4 rubric dimensions using an LLM judge.

    Falls back to neutral scores (0.5) if LLM call fails.

    Parameters
    ----------
    mep : dict
        Parsed MEP dict.
    model : str
        Gemini model to use for judging.

    Returns
    -------
    dict
        Scores for rationale_specificity, constraint_grounding,
        decision_logic, no_fabrication, plus judge_notes.
    """
    from ..utils.json_strict import parse_strict

    gov = (mep.get("governance") or {}).get("parsed", {})
    bm = (mep.get("benchmark") or {}).get("parsed", {})
    agg = (mep.get("aggregator") or {}).get("parsed", {})

    outputs = {
        "governance_decision": gov.get("decision"),
        "governance_flags": gov.get("flags", []),
        "governance_rationale": gov.get("rationale", ""),
        "governance_metrics": gov.get("metrics", {}),
        "benchmark_decision": bm.get("decision"),
        "benchmark_comparisons": bm.get("comparisons", {}),
        "benchmark_rationale": bm.get("rationale", ""),
        "final_decision": agg.get("final_decision"),
        "aggregator_rationale": agg.get("rationale", ""),
        "aggregator_overrides": agg.get("overrides", []),
    }

    prompt = JUDGE_PROMPT.replace("{outputs_json}", json.dumps(outputs, indent=2))

    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return _neutral_scores("No GEMINI_API_KEY set")

        from crewai import LLM, Agent, Crew, Task  # type: ignore

        llm = LLM(model=f"gemini/{model}", api_key=api_key, temperature=0)
        agent = Agent(
            role="Audit Evaluator",
            goal="Score the pipeline output on 4 rubric dimensions. Output JSON only.",
            backstory="You are a rigorous compliance auditor.",
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=prompt,
            expected_output="JSON: {rationale_specificity, constraint_grounding, decision_logic, no_fabrication, judge_notes}",
            agent=agent,
        )
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        raw = getattr(result, "raw", None) or str(result)
        parsed, ok = parse_strict(
            raw,
            required_keys=["rationale_specificity", "constraint_grounding", "decision_logic", "no_fabrication"],
        )
        if ok:
            return parsed
        return _neutral_scores("Judge parse failed")

    except Exception as exc:
        return _neutral_scores(f"Judge error: {exc}")


def _neutral_scores(reason: str = "") -> dict:
    """Return neutral 0.5 scores when judge cannot run."""
    return {
        "rationale_specificity": 0.5,
        "constraint_grounding": 0.5,
        "decision_logic": 0.5,
        "no_fabrication": 0.5,
        "judge_notes": reason or "Neutral fallback — judge did not run.",
    }
