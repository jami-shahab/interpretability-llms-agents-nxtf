"""CapitalCase — the canonical data container for one NXTFrontier capital proposal."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Canonical decision labels
VALID_DECISIONS = {"PROCEED", "DECLINE", "ESCALATE", "PROCEED_WITH_MITIGATION"}
VALID_LENS_DECISIONS = {"PASS", "FAIL", "ESCALATE", "unanswerable"}


@dataclass
class CapitalCase:
    """Standardized container for one capital allocation proposal.

    Attributes
    ----------
    case_id : str
        Unique identifier, e.g. ``"CASE_01"``.
    company_profile : dict
        Industry, asset_archetype, revenue, ebitda_margin, etc.
    expansion_plan : dict
        Capex, strategic_rationale, risk_factors, ROI timeline.
    financials : dict
        Balance sheet (debt, cash) and financial ratios.
    financing_details : dict
        Funding sources and government programme flags.
    approval_context : dict
        Escalation thresholds (IC, board).
    constraint_formalization : list
        Typed constraint rules (C1–C5 + structured_limits).
    external_context : dict
        Interest rates, industry growth rate.
    planner_answerability : dict
        Per-lens answerability flags from ground truth.
    expected_outcome : dict
        Ground-truth: final_decision + per_lens decisions.
    eval_metrics : dict
        must_cite, must_flag, must_compare checklists.
    metadata : dict
        Annotator, date, confidentiality, etc.
    """

    case_id: str
    company_profile: Dict[str, Any]
    expansion_plan: Dict[str, Any]
    financials: Dict[str, Any]
    financing_details: Dict[str, Any]
    approval_context: Dict[str, Any]
    constraint_formalization: List[Dict[str, Any]]
    external_context: Dict[str, Any]
    planner_answerability: Dict[str, Any]
    expected_outcome: Dict[str, Any]
    eval_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Convenience accessors (avoids deep dict access in agent code)
    # ------------------------------------------------------------------ #

    def ebitda(self) -> float:
        """Compute EBITDA from revenue × margin."""
        return (
            self.company_profile.get("revenue", 0)
            * self.company_profile.get("ebitda_margin", 0)
        )

    def asset_archetype(self) -> str:
        """Return the asset archetype string, e.g. ``'A_Fixed_Infrastructure'``."""
        return self.company_profile.get("asset_archetype", "")

    def required_capex(self) -> float:
        """Return the proposed capital expenditure in CAD."""
        return float(self.expansion_plan.get("required_capex", 0))

    def current_debt(self) -> float:
        """Return current total debt from balance sheet."""
        return float(self.financials.get("balance_sheet", {}).get("debt", 0))

    def current_cash(self) -> float:
        """Return current cash from balance sheet."""
        return float(self.financials.get("balance_sheet", {}).get("cash", 0))

    def expected_final_decision(self) -> str:
        """Return the ground-truth final decision label."""
        return self.expected_outcome.get("final_decision", "")

    def expected_per_lens(self) -> Dict[str, str]:
        """Return the per-lens expected decisions {sponsor, governance, benchmark}."""
        return self.expected_outcome.get("per_lens_expected", {})

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Serialise to a plain dict for MEP embedding."""
        return {
            "case_id": self.case_id,
            "company_profile": self.company_profile,
            "expansion_plan": self.expansion_plan,
            "financials": self.financials,
            "financing_details": self.financing_details,
            "approval_context": self.approval_context,
            "constraint_formalization": self.constraint_formalization,
            "external_context": self.external_context,
            "planner_answerability": self.planner_answerability,
            "expected_outcome": self.expected_outcome,
            "eval_metrics": self.eval_metrics,
            "metadata": self.metadata,
        }

    # ------------------------------------------------------------------ #
    # Agent context projections — slim views sent to each specific agent
    # ------------------------------------------------------------------ #

    def sponsor_context(self) -> Dict[str, Any]:
        """Return only the fields the Sponsor Agent needs (project economics)."""
        return {
            "case_id": self.case_id,
            "industry": self.company_profile.get("industry"),
            "asset_archetype": self.asset_archetype(),
            "required_capex": self.required_capex(),
            "expected_revenue_uplift": self.expansion_plan.get("expected_revenue_uplift", 0),
            "time_to_roi_years": self.expansion_plan.get("time_to_roi_years"),
            "strategic_rationale": self.expansion_plan.get("strategic_rationale", ""),
            "risk_factors": self.expansion_plan.get("risk_factors", []),
        }

    def governance_context(self) -> Dict[str, Any]:
        """Return only the fields the Governance Agent needs (balance sheet + policy)."""
        structured = next(
            (c for c in self.constraint_formalization if c.get("id") == "structured_limits"),
            {},
        )
        return {
            "case_id": self.case_id,
            "current_debt": self.current_debt(),
            "current_cash": self.current_cash(),
            "ebitda": self.ebitda(),
            "current_debt_to_ebitda": self.financials.get("ratios", {}).get("debt_to_ebitda"),
            "required_capex": self.required_capex(),
            "financing_sources": self.financing_details.get("sources", []),
            "government_programs": self.financing_details.get("government_programs", {}),
            "asset_archetype": self.asset_archetype(),
            "strategic_rationale": self.expansion_plan.get("strategic_rationale", ""),
            "approval_context": self.approval_context,
            "policy_limits": structured,
        }

    def benchmark_context(self) -> Dict[str, Any]:
        """Return only the fields the Benchmark Agent needs (industry + external)."""
        return {
            "case_id": self.case_id,
            "asset_archetype": self.asset_archetype(),
            "industry": self.company_profile.get("industry"),
            "required_capex": self.required_capex(),
            "timeline_quarters": self.expansion_plan.get("timeline_quarters"),
            "external_context": self.external_context,
            "current_fleet_utilization": self.expansion_plan.get("fleet_utilization"),
            "strategic_rationale": self.expansion_plan.get("strategic_rationale", ""),
            "risk_factors": self.expansion_plan.get("risk_factors", []),
        }

    def aggregator_context(
        self,
        sponsor_output: Optional[Dict] = None,
        governance_output: Optional[Dict] = None,
        benchmark_output: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Return the slim aggregator view: the three sub-agent reports only."""
        return {
            "case_id": self.case_id,
            "required_capex": self.required_capex(),
            "sponsor": sponsor_output or {},
            "governance": governance_output or {},
            "benchmark": benchmark_output or {},
        }
