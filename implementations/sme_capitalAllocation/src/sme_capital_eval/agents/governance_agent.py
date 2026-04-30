"""GovernanceAgent — enforces fund-level policies (Treasury Policy + SAMP).

Receives only the governance projection of CapitalCase.
Uses FinancialCalcTool (arithmetic) + PolicyRetrievalTool (policy text).
"""

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from crewai import Agent, Crew, Task
from dotenv import load_dotenv

from ..tools.financial_calc_tool import FinancialCalcTool
from ..tools.policy_retrieval_tool import InMemoryPolicyAdapter, PolicyRetrievalTool
from ..utils.json_strict import parse_strict
from ..utils.llm_adapter import GeminiLLMAdapter, LLMAdapterPort


load_dotenv()

_PROMPT_PATH = Path(__file__).parent / "prompts" / "governance.txt"
_REQUIRED_KEYS = ["agent", "decision", "flags", "metrics", "rationale"]



class GovernanceAgent:
    """Enforces internal covenants: leverage ceiling, liquidity floor, SAMP, IC authority.

    Decision logic is primarily deterministic — the LLM is used only to
    reason over retrieved policy text and format the output JSON.
    """

    def __init__(
        self,
        llm_adapter: Optional[LLMAdapterPort] = None,
        policy_adapter: Optional[Any] = None,
    ) -> None:
        self._adapter = llm_adapter or GeminiLLMAdapter()
        self._llm = self._adapter.build()
        self._template = _PROMPT_PATH.read_text()
        self._calc_tool = FinancialCalcTool()
        policy = policy_adapter or InMemoryPolicyAdapter()
        self._policy_tool = PolicyRetrievalTool(adapter=policy)

    def _build_prompt(self, slim_context: dict) -> str:
        return self._template.replace("{proposal_json}", json.dumps(slim_context, indent=2))

    def run(self, slim_context: dict, lf_trace: Any = None) -> Tuple[str, dict, bool, str, List]:
        """Evaluate governance lens.

        Parameters
        ----------
        slim_context : dict
            Output of ``CapitalCase.governance_context()``.

        Returns
        -------
        prompt, parsed, parse_error, raw_text, tool_trace
        """
        prompt = self._build_prompt(slim_context)

        agent = Agent(
            role="Corporate Oversight Guardian",
            goal="Enforce NXTFrontier policies. Flag every violation. Output JSON only.",
            backstory="You are a CPA-grade financial controller protecting the fund's balance sheet.",
            llm=self._llm,
            tools=[self._calc_tool, self._policy_tool],
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=prompt,
            expected_output="JSON: {agent, decision, flags, metrics, rationale}",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        raw_text: str = getattr(result, "raw", None) or str(result)
        parsed, ok = parse_strict(raw_text, required_keys=_REQUIRED_KEYS)

        if not parsed:
            parsed = {
                "agent": "governance",
                "decision": "unanswerable",
                "flags": [],
                "metrics": {},
                "rationale": "Governance parse failed.",
            }

        return prompt, parsed, not ok, raw_text, []
