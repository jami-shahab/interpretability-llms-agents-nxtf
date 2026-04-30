"""BenchmarkAgent — external market reality check using StatsCan data.

Receives only the benchmark projection of CapitalCase + NAICS tags from Planner.
Uses StatCanQueryTool for deterministic lookups — LLM only reasons over retrieved numbers.
"""

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from crewai import Agent, Crew, Task
from dotenv import load_dotenv

from ..tools.statscan_query_tool import SQLiteBenchmarkAdapter, StatCanQueryTool
from ..utils.json_strict import parse_strict
from ..utils.llm_adapter import GeminiLLMAdapter, LLMAdapterPort


load_dotenv()

_PROMPT_PATH = Path(__file__).parent / "prompts" / "benchmark.txt"
_REQUIRED_KEYS = ["agent", "decision", "flags", "comparisons", "rationale"]



class BenchmarkAgent:
    """Validates proposal against StatsCan capex norms, financial ratios, and SFGSME lending data.

    The LLM never fabricates benchmark numbers — it only reasons over
    numbers fetched by StatCanQueryTool from the processed SQLite database.
    """

    def __init__(
        self,
        llm_adapter: Optional[LLMAdapterPort] = None,
        benchmark_adapter: Optional[Any] = None,
    ) -> None:
        self._adapter = llm_adapter or GeminiLLMAdapter()
        self._llm = self._adapter.build()
        self._template = _PROMPT_PATH.read_text()
        bench = benchmark_adapter or SQLiteBenchmarkAdapter()
        self._query_tool = StatCanQueryTool(adapter=bench)

    def _build_prompt(
        self,
        slim_context: dict,
        enterprise_naics: str,
        project_naics: str,
    ) -> str:
        return (
            self._template
            .replace("{enterprise_naics}", enterprise_naics)
            .replace("{project_naics}", project_naics)
            .replace("{proposal_json}", json.dumps(slim_context, indent=2))
        )

    def run(
        self,
        slim_context: dict,
        plan_parsed: dict,
        lf_trace: Any = None,
    ) -> Tuple[str, dict, bool, str, List]:
        """Evaluate benchmark lens.

        Parameters
        ----------
        slim_context : dict
            Output of ``CapitalCase.benchmark_context()``.
        plan_parsed : dict
            Parsed output of PlannerAgent (contains enterprise_naics, project_naics).

        Returns
        -------
        prompt, parsed, parse_error, raw_text, tool_trace
        """
        enterprise_naics = plan_parsed.get("enterprise_naics", "00")
        project_naics = plan_parsed.get("project_naics", "00")
        prompt = self._build_prompt(slim_context, enterprise_naics, project_naics)

        agent = Agent(
            role="Industry Benchmark Appraiser",
            goal="Validate proposal against real market data. Output JSON only.",
            backstory=(
                "You are an independent market appraiser who only uses numbers retrieved "
                "from the statscan_query_tool — never guessing or hallucinating benchmarks."
            ),
            llm=self._llm,
            tools=[self._query_tool],
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=prompt,
            expected_output="JSON: {agent, decision, flags, comparisons, rationale}",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        raw_text: str = getattr(result, "raw", None) or str(result)
        parsed, ok = parse_strict(raw_text, required_keys=_REQUIRED_KEYS)

        if not parsed:
            parsed = {
                "agent": "benchmark",
                "decision": "unanswerable",
                "flags": [],
                "comparisons": {},
                "rationale": "Benchmark parse failed.",
            }

        return prompt, parsed, not ok, raw_text, []
