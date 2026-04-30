r"""Runner: generate Model Evaluation Packets for SME Capital Allocation cases.

Usage (Gemini default):
    uv run --env-file <.env> --directory <sme_capitalAllocation> \
        -m sme_capital_eval.runner.run_generate_meps \
        --split test --n 7 --workers 1 --out meps/

Usage (Ollama):
    uv run --directory <sme_capitalAllocation> \
        -m sme_capital_eval.runner.run_generate_meps \
        --provider ollama --model gemma4:e2b \
        --ollama_url http://localhost:11434 \
        --split test --n 7 --workers 1 --out meps/
"""

import argparse
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ..agents.aggregator_agent import AggregatorAgent
from ..agents.benchmark_agent import BenchmarkAgent
from ..agents.governance_agent import GovernanceAgent
from ..agents.planner_agent import PlannerAgent
from ..agents.sponsor_agent import SponsorAgent
from ..datasets.capital_case import CapitalCase
from ..datasets.case_loader import load_cases
from ..langfuse_integration.client import get_client
from ..langfuse_integration.tracing import log_trace_scores, sample_trace
from ..mep.schema import (
    MEP,
    MEPAgentOutput,
    MEPAggregator,
    MEPCase,
    MEPConfig,
    MEPPlan,
    MEPTimestamps,
)
from ..mep.writer import write_mep
from ..utils.llm_adapter import LLMAdapterPort, build_adapter
from ..utils.timing import iso_now, timed


load_dotenv()


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------


def process_case(
    case: CapitalCase,
    planner: PlannerAgent,
    sponsor: SponsorAgent,
    governance: GovernanceAgent,
    benchmark: BenchmarkAgent,
    aggregator: AggregatorAgent,
    config: dict,
    run_id: str,
    out_dir: str,
    lf_client=None,
    workers: int = 1,
) -> str:
    """Run the full 5-stage pipeline for a single capital case.

    Parameters
    ----------
    case : CapitalCase
        The proposal to evaluate.
    planner, sponsor, governance, benchmark, aggregator : Agent instances
    config : dict
        Run configuration (model names, config_name).
    run_id : str
        Shared UUID for this evaluation run.
    out_dir : str
        MEP output directory.
    lf_client : optional
        Langfuse client.
    workers : int
        Number of threads for parallel middle-agent execution.

    Returns
    -------
    str
        Path to the written MEP file.
    """
    run_start = iso_now()
    errors: list = []

    with sample_trace(
        lf_client,
        case_id=case.case_id,
        expected_decision=case.expected_final_decision(),
        config_name=config["config_name"],
        run_id=run_id,
    ) as lf_trace:
        lf_trace_id = getattr(lf_trace, "id", None)

        # ---- Stage 1: Planner ----
        plan_prompt, plan_parsed, plan_err, plan_raw = "", {}, True, ""
        plan_ms = 0.0
        try:
            with timed() as t:
                plan_prompt, plan_parsed, plan_err, plan_raw = planner.run(
                    slim_context={
                        "case_id": case.case_id,
                        "industry": case.company_profile.get("industry"),
                        "asset_archetype": case.asset_archetype(),
                        "required_capex": case.required_capex(),
                        "strategic_rationale": case.expansion_plan.get("strategic_rationale", ""),
                        "risk_factors": case.expansion_plan.get("risk_factors", []),
                    },
                    lf_trace=lf_trace,
                )
            plan_ms = t.elapsed_ms
        except Exception as exc:
            errors.append(f"planner_error: {exc}")
            traceback.print_exc()

        # ---- Stages 2–4: Sponsor, Governance, Benchmark (parallel or sequential) ----
        sp_prompt, sp_parsed, sp_err, sp_raw, sp_traces = "", {}, True, "", []
        gov_prompt, gov_parsed, gov_err, gov_raw, gov_traces = "", {}, True, "", []
        bm_prompt, bm_parsed, bm_err, bm_raw, bm_traces = "", {}, True, "", []
        sp_ms = gov_ms = bm_ms = 0.0

        def _run_sponsor():
            with timed() as t:
                out = sponsor.run(case.sponsor_context(), lf_trace)
            return out, t.elapsed_ms

        def _run_governance():
            with timed() as t:
                out = governance.run(case.governance_context(), lf_trace)
            return out, t.elapsed_ms

        def _run_benchmark():
            with timed() as t:
                out = benchmark.run(case.benchmark_context(), plan_parsed, lf_trace)
            return out, t.elapsed_ms

        if workers > 1:
            jobs = {
                "sponsor": _run_sponsor,
                "governance": _run_governance,
                "benchmark": _run_benchmark,
            }
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(fn): name for name, fn in jobs.items()}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result_tuple, elapsed = future.result()
                        if name == "sponsor":
                            sp_prompt, sp_parsed, sp_err, sp_raw, sp_traces = result_tuple
                            sp_ms = elapsed
                        elif name == "governance":
                            gov_prompt, gov_parsed, gov_err, gov_raw, gov_traces = result_tuple
                            gov_ms = elapsed
                        else:
                            bm_prompt, bm_parsed, bm_err, bm_raw, bm_traces = result_tuple
                            bm_ms = elapsed
                    except Exception as exc:
                        errors.append(f"{name}_error: {exc}")
                        traceback.print_exc()
        else:
            # Sequential (safer for rate-limited free plan)
            try:
                (sp_prompt, sp_parsed, sp_err, sp_raw, sp_traces), sp_ms = _run_sponsor()
            except Exception as exc:
                errors.append(f"sponsor_error: {exc}")
                traceback.print_exc()
            try:
                (gov_prompt, gov_parsed, gov_err, gov_raw, gov_traces), gov_ms = _run_governance()
            except Exception as exc:
                errors.append(f"governance_error: {exc}")
                traceback.print_exc()
            try:
                (bm_prompt, bm_parsed, bm_err, bm_raw, bm_traces), bm_ms = _run_benchmark()
            except Exception as exc:
                errors.append(f"benchmark_error: {exc}")
                traceback.print_exc()

        # ---- Stage 5: Aggregator ----
        agg_prompt, agg_parsed, agg_err, agg_raw = "", {}, True, ""
        rule_decision, rule_applied = "ESCALATE", "fallback"
        agg_ms = 0.0
        try:
            with timed() as t:
                (
                    agg_prompt,
                    agg_parsed,
                    agg_err,
                    agg_raw,
                    rule_decision,
                    rule_applied,
                ) = aggregator.run(
                    aggregator_context=case.aggregator_context(sp_parsed, gov_parsed, bm_parsed),
                    sponsor_parsed=sp_parsed,
                    governance_parsed=gov_parsed,
                    benchmark_parsed=bm_parsed,
                    lf_trace=lf_trace,
                )
            agg_ms = t.elapsed_ms
        except Exception as exc:
            errors.append(f"aggregator_error: {exc}")
            traceback.print_exc()

        run_end = iso_now()

        # ---- Assemble MEP ----
        mep = MEP(
            run_id=run_id,
            config=MEPConfig(
                planner_model=config["model"],
                agent_model=config["model"],
                config_name=config["config_name"],
            ),
            case=MEPCase(
                case_id=case.case_id,
                industry=case.company_profile.get("industry", ""),
                asset_archetype=case.asset_archetype(),
                required_capex=case.required_capex(),
                expected_final_decision=case.expected_final_decision(),
                expected_per_lens=case.expected_per_lens(),
                eval_metrics=case.eval_metrics,
                metadata=case.metadata,
            ),
            plan=MEPPlan(
                prompt=plan_prompt,
                raw_text=plan_raw,
                parsed=plan_parsed,
                parse_error=plan_err,
            ),
            sponsor=MEPAgentOutput(
                agent_name="sponsor",
                prompt=sp_prompt,
                raw_text=sp_raw,
                parsed=sp_parsed,
                parse_error=sp_err,
                tool_trace=sp_traces,
            ),
            governance=MEPAgentOutput(
                agent_name="governance",
                prompt=gov_prompt,
                raw_text=gov_raw,
                parsed=gov_parsed,
                parse_error=gov_err,
                tool_trace=gov_traces,
            ),
            benchmark=MEPAgentOutput(
                agent_name="benchmark",
                prompt=bm_prompt,
                raw_text=bm_raw,
                parsed=bm_parsed,
                parse_error=bm_err,
                tool_trace=bm_traces,
            ),
            aggregator=MEPAggregator(
                prompt=agg_prompt,
                raw_text=agg_raw,
                parsed=agg_parsed,
                parse_error=agg_err,
                rule_decision=rule_decision,
                rule_applied=rule_applied,
            ),
            timestamps=MEPTimestamps(
                start=run_start,
                end=run_end,
                planner_ms=plan_ms,
                sponsor_ms=sp_ms,
                governance_ms=gov_ms,
                benchmark_ms=bm_ms,
                aggregator_ms=agg_ms,
            ),
            errors=errors,
            lf_trace_id=lf_trace_id,
        )

        log_trace_scores(
            lf_trace,
            {
                "planner_parse_ok": float(not plan_err),
                "sponsor_parse_ok": float(not sp_err),
                "governance_parse_ok": float(not gov_err),
                "benchmark_parse_ok": float(not bm_err),
                "aggregator_parse_ok": float(not agg_err),
                "has_errors": float(bool(errors)),
            },
        )

    return write_mep(mep, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the MEP generation pipeline."""
    parser = argparse.ArgumentParser(description="Generate MEPs for SME Capital Allocation")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "ollama"],
                        help="LLM provider (default: gemini)")
    parser.add_argument("--model", default=None,
                        help="Model name. Defaults: gemini-2.0-flash-lite (Gemini), gemma4:e2b (Ollama)")
    parser.add_argument("--ollama_url", default="http://localhost:11434",
                        help="Ollama server URL (only used with --provider ollama)")
    parser.add_argument("--split", default="test", choices=["test", "fewshot", "all"])
    parser.add_argument("--n", type=int, default=None, help="Number of cases (default: all in split)")
    parser.add_argument("--workers", type=int, default=1,
                        help="1=sequential (recommended for free Gemini plan)")
    parser.add_argument("--out", default="meps/", help="Output directory for MEPs")
    args = parser.parse_args()

    # Resolve model default per provider
    model = args.model
    if model is None:
        model = "gemma4:e2b" if args.provider == "ollama" else "gemini-2.0-flash-lite"

    # Build the shared LLM adapter (injected into all 5 agents)
    llm_adapter: LLMAdapterPort = build_adapter(
        provider=args.provider,
        model=model,
        ollama_url=args.ollama_url,
    )

    safe_model = model.replace("-", "_").replace(".", "_").replace(":", "_")
    config = {
        "model": model,
        "provider": args.provider,
        "config_name": f"{args.provider}_{safe_model}",
    }
    run_id = str(uuid.uuid4())
    out_dir = str(Path(args.out) / config["config_name"] / args.split)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading cases   : split={args.split}")
    cases = load_cases(split=args.split, n=args.n)
    print(f"Cases loaded    : {len(cases)}")
    print(f"Provider        : {args.provider}")
    print(f"Model           : {model}")
    if args.provider == "ollama":
        print(f"Ollama URL      : {args.ollama_url}")
    print(f"Workers         : {args.workers}  (use 1 on free Gemini plan)")
    print(f"Output dir      : {out_dir}")
    print()

    lf_client = get_client()
    print(f"Langfuse        : {'enabled' if lf_client else 'not configured'}")
    print()

    print("Initialising agents …")
    planner = PlannerAgent(llm_adapter=llm_adapter)
    sponsor = SponsorAgent(llm_adapter=llm_adapter)
    governance = GovernanceAgent(llm_adapter=llm_adapter)
    bm = BenchmarkAgent(llm_adapter=llm_adapter)
    agg = AggregatorAgent(llm_adapter=llm_adapter)

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.case_id} …", end=" ", flush=True)
        try:
            path = process_case(
                case, planner, sponsor, governance, bm, agg,
                config, run_id, out_dir, lf_client, args.workers,
            )
            print(f"OK \u2192 {path}")
        except Exception as exc:
            print(f"ERROR: {exc}")
            traceback.print_exc()

    print(f"\nDone. MEPs written to: {out_dir}")


if __name__ == "__main__":
    main()
