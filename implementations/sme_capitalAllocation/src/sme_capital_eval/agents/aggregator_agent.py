"""AggregatorAgent — the Governance Board / Investment Committee.

Hybrid architecture:
1. A deterministic rule engine (pure Python) sets ``final_decision`` based on sub-agent flags.
2. The LLM writes the rationale, overrides narrative, and required_actions.

This guarantees policy enforcement is never probabilistic while still producing
human-readable, explainable output.
"""

import json
from pathlib import Path
from typing import Any, Optional, Tuple

from crewai import Agent, Crew, Task
from dotenv import load_dotenv

from ..utils.json_strict import parse_strict
from ..utils.llm_adapter import GeminiLLMAdapter, LLMAdapterPort


load_dotenv()

_PROMPT_PATH = Path(__file__).parent / "prompts" / "aggregator.txt"
_REQUIRED_KEYS = ["agent", "final_decision", "decision_path", "overrides", "rationale"]



# ---------------------------------------------------------------------------
# Deterministic decision rule engine
# ---------------------------------------------------------------------------


def apply_decision_rules(
    sponsor: dict,
    governance: dict,
    benchmark: dict,
) -> Tuple[str, str]:
    """Apply the Investment Committee decision table in strict priority order.

    Parameters
    ----------
    sponsor : dict
        Parsed SponsorAgent output.
    governance : dict
        Parsed GovernanceAgent output.
    benchmark : dict
        Parsed BenchmarkAgent output.

    Returns
    -------
    final_decision : str
        One of PROCEED, DECLINE, ESCALATE, PROCEED_WITH_MITIGATION.
    rule_applied : str
        Human-readable description of the rule that fired.
    """
    gov_flags = set(governance.get("flags", []))
    bench_flags = set(benchmark.get("flags", []))
    sponsor_dec = sponsor.get("decision", "unanswerable")
    gov_dec = governance.get("decision", "unanswerable")
    bench_dec = benchmark.get("decision", "unanswerable")

    # Rule 1: All PASS
    if all(d == "PASS" for d in [sponsor_dec, gov_dec, bench_dec]):
        return "PROCEED", "Rule 1: Unanimous PASS across all three lenses"

    # Rule 2: Regulatory / permitting failure (external hard stop)
    if {"permitting_buffer_missing", "environmental_review_bypass"} & bench_flags:
        return "DECLINE", "Rule 2: Regulatory/Permitting Failure — mandatory external review missing"

    # Rule 3: Liquidity floor violation (hard stop — balance sheet ruin)
    if "cash_floor_violation" in gov_flags:
        return "DECLINE", "Rule 3: Liquidity Floor Violation — cash drops below $250k minimum"

    # Rule 4: SAMP violation (ISO 55000 lifecycle breach — hard stop)
    if {"SAMP_deferred_maintenance_violation", "SAMP_violation"} & gov_flags:
        return "DECLINE", "Rule 4: SAMP Violation — ISO 55000 asset lifecycle mandate breached"

    # Rule 5: Policy breach requiring board (leverage or IC authority)
    if {"leverage_ceiling_breach", "IC_authority_exceeded"} & gov_flags:
        return "ESCALATE", "Rule 5: Policy Breach — leverage >2.5x or capex exceeds IC authority ($2M)"

    # Rule 6: Operational data conflict
    if "conflicting_utilization_data" in bench_flags or bench_dec == "ESCALATE":
        return "ESCALATE", "Rule 6: Operational Data Conflict — utilization data is inconsistent"

    # Rule 7: Utilization below threshold (fleet archetype gate)
    if "utilization_below_threshold" in bench_flags:
        return "DECLINE", "Rule 7: Fleet Utilization Below 45% Threshold"

    # Rule 8: Financing limit breach without sub-debt blend
    if "financing_limit_breach" in gov_flags:
        return "ESCALATE", "Rule 8: Financing Structure Breach — single draw >$1.15M without BDC blend"

    # Rule 9: Sponsor PASS + fixable governance issue → proceed with mitigations
    if sponsor_dec == "PASS" and gov_dec == "FAIL":
        return "PROCEED_WITH_MITIGATION", "Rule 9: Strong ROI with fixable financing/governance structure"

    # Rule 10: Any unanswerable lens → escalate for human review
    if "unanswerable" in {sponsor_dec, gov_dec, bench_dec}:
        return "ESCALATE", "Rule 10: One or more lenses returned unanswerable — insufficient data"

    # Default fallback
    return "ESCALATE", "Rule 10: Unresolved conflict — requires human review"


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class AggregatorAgent:
    """Investment Committee: resolves adversarial sub-agent reports into a final decision.

    The rule engine enforces governance deterministically;
    the LLM writes the rationale that makes it auditable and human-readable.
    """

    def __init__(self, llm_adapter: Optional[LLMAdapterPort] = None) -> None:
        self._adapter = llm_adapter or GeminiLLMAdapter()
        self._llm = self._adapter.build()
        self._template = _PROMPT_PATH.read_text()

    def _build_prompt(
        self,
        aggregator_context: dict,
        rule_decision: str,
        rule_applied: str,
    ) -> str:
        reports = {
            "sponsor": aggregator_context.get("sponsor", {}),
            "governance": aggregator_context.get("governance", {}),
            "benchmark": aggregator_context.get("benchmark", {}),
        }
        return (
            self._template
            .replace("{rule_decision}", rule_decision)
            .replace("{rule_applied}", rule_applied)
            .replace("{reports_json}", json.dumps(reports, indent=2))
        )

    def run(
        self,
        aggregator_context: dict,
        sponsor_parsed: dict,
        governance_parsed: dict,
        benchmark_parsed: dict,
        lf_trace: Any = None,
    ) -> Tuple[str, dict, bool, str, str, str]:
        """Run the aggregator.

        Parameters
        ----------
        aggregator_context : dict
            Output of ``CapitalCase.aggregator_context(...)``.
        sponsor_parsed, governance_parsed, benchmark_parsed : dict
            Parsed outputs from the three sub-agents.

        Returns
        -------
        prompt, parsed, parse_error, raw_text, rule_decision, rule_applied
        """
        # Step 1: deterministic rules
        rule_decision, rule_applied = apply_decision_rules(
            sponsor_parsed, governance_parsed, benchmark_parsed
        )

        # Step 2: LLM writes rationale
        prompt = self._build_prompt(aggregator_context, rule_decision, rule_applied)

        agent = Agent(
            role="Governance Board Investment Committee",
            goal="Write a clear rationale for the pre-determined decision. Output JSON only.",
            backstory="You are the CEO, CFO, and CCO collectively making the final capital decision.",
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=prompt,
            expected_output="JSON: {agent, final_decision, decision_path, overrides, rationale, required_actions}",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        raw_text: str = getattr(result, "raw", None) or str(result)
        parsed, ok = parse_strict(raw_text, required_keys=_REQUIRED_KEYS)

        # Enforce rule engine's decision (LLM cannot override)
        if parsed:
            parsed["final_decision"] = rule_decision
            parsed["decision_path"] = rule_applied
        else:
            parsed = {
                "agent": "verifier",
                "final_decision": rule_decision,
                "decision_path": rule_applied,
                "overrides": [],
                "rationale": f"Aggregator parse failed. Rule engine determined: {rule_decision}.",
                "required_actions": [],
            }

        return prompt, parsed, not ok, raw_text, rule_decision, rule_applied
