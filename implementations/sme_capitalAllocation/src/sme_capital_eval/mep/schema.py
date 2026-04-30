"""MEP schema for the SME Capital Allocation pipeline.

``mep.capital.v1`` stores a complete, replayable trace of one pipeline run:
Planner → [Sponsor, Governance, Benchmark] → Aggregator.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MEPConfig:
    """LLM backend configuration used for this run."""

    planner_model: str
    agent_model: str
    config_name: str  # e.g. "gemini_gemini"


@dataclass
class MEPCase:
    """Minimal case metadata embedded in the MEP for self-contained traceability."""

    case_id: str
    industry: str
    asset_archetype: str
    required_capex: float
    expected_final_decision: str
    expected_per_lens: Dict[str, str]
    eval_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MEPPlan:
    """Planner agent output: NAICS tags + per-lens answerability."""

    prompt: str
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)
    # Expected keys: enterprise_naics, project_naics, answerability {sponsor, governance, benchmark}
    parse_error: bool = False


@dataclass
class MEPAgentOutput:
    """Generic container for one sub-agent report (Sponsor / Governance / Benchmark)."""

    agent_name: str  # "sponsor" | "governance" | "benchmark"
    prompt: str
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)
    # Sponsor: {decision, key_metrics, rationale}
    # Governance: {decision, flags, metrics, rationale}
    # Benchmark: {decision, flags, comparisons, rationale}
    parse_error: bool = False
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MEPAggregator:
    """Aggregator (Governance Board) output: final binding decision."""

    prompt: str
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)
    # Keys: final_decision, decision_path, overrides, rationale, required_actions
    parse_error: bool = False
    # The deterministic rule outcome (set by rule engine before LLM rationale)
    rule_decision: str = ""
    rule_applied: str = ""


@dataclass
class MEPTimestamps:
    """Wall-clock latency per pipeline stage (milliseconds)."""

    start: str
    end: str
    planner_ms: float = 0.0
    sponsor_ms: float = 0.0
    governance_ms: float = 0.0
    benchmark_ms: float = 0.0
    aggregator_ms: float = 0.0


@dataclass
class MEP:
    """Complete Model Evaluation Packet for one capital allocation case."""

    schema_version: str = "mep.capital.v1"
    run_id: str = ""
    config: Optional[MEPConfig] = None
    case: Optional[MEPCase] = None
    plan: Optional[MEPPlan] = None
    sponsor: Optional[MEPAgentOutput] = None
    governance: Optional[MEPAgentOutput] = None
    benchmark: Optional[MEPAgentOutput] = None
    aggregator: Optional[MEPAggregator] = None
    timestamps: Optional[MEPTimestamps] = None
    errors: List[str] = field(default_factory=list)
    lf_trace_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict representation."""
        return dataclasses.asdict(self)
