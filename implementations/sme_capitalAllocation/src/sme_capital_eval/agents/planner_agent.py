"""PlannerAgent — tags the capital proposal with NAICS codes and answerability.

Text-only LLM; receives a slim projection of CapitalCase.
LLM provider is injected via LLMAdapterPort — swap Gemini ↔ Ollama with zero
changes to this file.
"""

import json
from pathlib import Path
from typing import Any, Optional, Tuple

from crewai import Agent, Crew, Task
from dotenv import load_dotenv

from ..utils.json_strict import parse_strict
from ..utils.llm_adapter import GeminiLLMAdapter, LLMAdapterPort
from ..utils.timing import iso_now


load_dotenv()

_PROMPT_PATH = Path(__file__).parent / "prompts" / "planner.txt"
_REQUIRED_KEYS = ["enterprise_naics", "project_naics", "answerability"]



class PlannerAgent:
    """Semantic tagger: routes the proposal to the correct NAICS benchmarks.

    Produces a compact JSON plan used by the Benchmark Agent for StatsCan lookups
    and signals answerability so the pipeline can skip unanswerable lenses.
    """

    def __init__(self, llm_adapter: Optional[LLMAdapterPort] = None) -> None:
        """Initialise the planner.

        Parameters
        ----------
        llm_adapter : LLMAdapterPort, optional
            LLM provider adapter.  Defaults to ``GeminiLLMAdapter`` using
            ``GEMINI_API_KEY`` from the environment.
        """
        self._adapter = llm_adapter or GeminiLLMAdapter()
        self._llm = self._adapter.build()
        self._template = _PROMPT_PATH.read_text()

    def _build_prompt(self, slim_context: dict) -> str:
        return self._template.replace("{proposal_json}", json.dumps(slim_context, indent=2))

    def run(
        self,
        slim_context: dict,
        lf_trace: Any = None,
    ) -> Tuple[str, dict, bool, str]:
        """Run the planner on a slim proposal context.

        Parameters
        ----------
        slim_context : dict
            Output of ``CapitalCase.sponsor_context()`` or a manually
            assembled dict with ``industry``, ``asset_archetype``,
            ``required_capex``, ``strategic_rationale``.
        lf_trace : Any, optional
            Langfuse trace handle (no-op if None).

        Returns
        -------
        prompt : str
        parsed : dict
        parse_error : bool
        raw_text : str
        """
        prompt = self._build_prompt(slim_context)

        agent = Agent(
            role="Capital Proposal Router",
            goal="Tag the proposal with NAICS codes and answerability. Output JSON only.",
            backstory="You are a precise intake analyst for an investment committee.",
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=prompt,
            expected_output="JSON with enterprise_naics, project_naics, answerability, routing_notes",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        raw_text: str = getattr(result, "raw", None) or str(result)
        parsed, ok = parse_strict(raw_text, required_keys=_REQUIRED_KEYS)

        # Fallback defaults
        if not parsed:
            parsed = {
                "enterprise_naics": "00",
                "project_naics": "00",
                "answerability": {
                    "sponsor": "answerable",
                    "governance": "answerable",
                    "benchmark": "answerable",
                },
                "routing_notes": "Planner parse failed — using defaults.",
            }

        return prompt, parsed, not ok, raw_text
