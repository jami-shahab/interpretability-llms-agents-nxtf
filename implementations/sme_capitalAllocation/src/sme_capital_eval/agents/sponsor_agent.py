"""SponsorAgent — evaluates project-level economics.

Receives only the expansion plan projection of CapitalCase.
Uses FinancialCalcTool for all arithmetic (never computes in tokens).
"""

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from crewai import Agent, Crew, Task
from dotenv import load_dotenv

from ..tools.financial_calc_tool import FinancialCalcTool
from ..utils.json_strict import parse_strict
from ..utils.llm_adapter import GeminiLLMAdapter, LLMAdapterPort


load_dotenv()

_PROMPT_PATH = Path(__file__).parent / "prompts" / "sponsor.txt"
_REQUIRED_KEYS = ["agent", "decision", "key_metrics", "rationale"]



class SponsorAgent:
    """Champions the project based on ROI and payback period.

    Always biased toward PASS — counterbalanced by GovernanceAgent.
    """

    def __init__(self, llm_adapter: Optional[LLMAdapterPort] = None) -> None:
        self._adapter = llm_adapter or GeminiLLMAdapter()
        self._llm = self._adapter.build()
        self._template = _PROMPT_PATH.read_text()
        self._calc_tool = FinancialCalcTool()

    def _build_prompt(self, slim_context: dict) -> str:
        return self._template.replace("{proposal_json}", json.dumps(slim_context, indent=2))

    def run(self, slim_context: dict, lf_trace: Any = None) -> Tuple[str, dict, bool, str, List]:
        """Evaluate sponsor lens.

        Parameters
        ----------
        slim_context : dict
            Output of ``CapitalCase.sponsor_context()``.

        Returns
        -------
        prompt, parsed, parse_error, raw_text, tool_trace
        """
        prompt = self._build_prompt(slim_context)

        agent = Agent(
            role="Project Sponsor",
            goal="Evaluate project economics. Output JSON only.",
            backstory="You are an aggressive, growth-focused capital allocator championing project ROI.",
            llm=self._llm,
            tools=[self._calc_tool],
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=prompt,
            expected_output="JSON: {agent, decision, key_metrics, rationale}",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        raw_text: str = getattr(result, "raw", None) or str(result)
        parsed, ok = parse_strict(raw_text, required_keys=_REQUIRED_KEYS)

        if not parsed:
            parsed = {
                "agent": "sponsor",
                "decision": "unanswerable",
                "key_metrics": {},
                "rationale": "Sponsor parse failed.",
            }

        return prompt, parsed, not ok, raw_text, []
