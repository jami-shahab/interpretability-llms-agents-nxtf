"""Unified Evaluation Pipeline Script.

This script executes the entire end-to-end SME Capital Allocation Agentic Evaluation
Framework in a single command. It chains:
  1. Generation of isolated, timestamped MEPs
  2. Scoring and output metric generation
  3. LLM trace and tool-use citation validation
  4. Decision error taxonomy classification
  5. Statistical summary compilation (printed to console and written to CSV)

Usage:
    uv run --env-file <.env> --directory <sme_capitalAllocation> \
        -m sme_capital_eval.runner.run_pipeline \
        --provider gemini --model gemini-2.5-flash
"""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list, cwd: Path, desc: str) -> bool:
    """Run a terminal command, stream stdout, and check return code."""
    print(f"\n>>> Running: {desc} ...")
    print(f"    Command: {' '.join(cmd)}")
    try:
        # Run subprocess and stream output directly to terminal
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR during {desc}: Command returned non-zero exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR executing {desc}: {e}")
        return False


def main() -> None:
    # 1. Parse arguments (same defaults as individual scripts for seamless integration)
    parser = argparse.ArgumentParser(description="Run the entire SME Capital Allocation Evaluation Suite")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai", "ollama"],
                        help="LLM provider (default: gemini)")
    parser.add_argument("--model", default=None,
                        help="Specific model string. Defaults: gemini-2.5-flash / gpt-4o-mini / gemma4:e2b")
    parser.add_argument("--split", type=str, default="test1", choices=["test1", "test2", "test3", "fewshot", "all"],
                        help="Cases split to run (default: test)")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of cases to evaluate (default: all in split)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrence workers (default: 1 for Gemini free tier)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Pacing delay between cases in seconds (e.g. 5.0 for Gemini free tier)")
    args = parser.parse_args()

    # Resolve default model per provider
    model = args.model
    if model is None:
        defaults = {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini", "ollama": "gemma4:e2b"}
        model = defaults[args.provider]

    # Resolve paths (relative to implementations/sme_capitalAllocation)
    # File is located at implementations/sme_capitalAllocation/src/sme_capital_eval/runner/run_pipeline.py
    project_dir = Path(__file__).parents[3].resolve()
    
    # 2. Determine target timestamped directory beforehand to align downstream evaluations
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("-", "_").replace(".", "_").replace(":", "_")
    config_name = f"{args.provider}_{safe_model}"
    target_split_folder = f"{args.split}_{now_str}"
    
    # Target absolute directories
    out_dir_base = project_dir / "meps"
    target_mep_dir = out_dir_base / config_name / target_split_folder
    
    print("=" * 60)
    print("SME CAPITAL ALLOCATION UNIFIED PIPELINE AUTOMATION")
    print("=" * 60)
    print(f"Project Workspace : {project_dir}")
    print(f"Provider          : {args.provider}")
    print(f"Model             : {model}")
    print(f"Split / Count     : {args.split} (n={args.n if args.n else 'all'})")
    print(f"Workers / Delay   : {args.workers} worker(s) / {args.delay}s delay")
    print(f"Target MEP Folder : meps/{config_name}/{target_split_folder}")
    print("=" * 60)

    # 3. Executable commands (uses python3 to run modules under the current venv context)
    python_bin = sys.executable

    # A. Generate MEPs
    gen_cmd = [
        python_bin, "-m", "sme_capital_eval.runner.run_generate_meps",
        "--provider", args.provider,
        "--model", model,
        "--split", args.split,
        "--workers", str(args.workers),
        "--delay", str(args.delay),
        "--out", "meps/",
        "--run_name", target_split_folder
    ]
    if args.n is not None:
        gen_cmd.extend(["--n", str(args.n)])

    if not run_cmd(gen_cmd, project_dir, "Stage 1: MEP Generation Pipeline"):
        sys.exit(1)

    # B. Score Decisions (eval_outputs)
    eval_out_file = project_dir / "output" / "metrics.jsonl"
    eval_cmd = [
        python_bin, "-m", "sme_capital_eval.eval.eval_outputs",
        "--mep_dir", str(target_mep_dir),
        "--out", str(eval_out_file)
    ]
    if not run_cmd(eval_cmd, project_dir, "Stage 2: Scoring Decision Accuracy"):
        sys.exit(1)

    # C. Score Traces (eval_traces)
    trace_out_file = project_dir / "output" / "trace_metrics.jsonl"
    trace_cmd = [
        python_bin, "-m", "sme_capital_eval.eval.eval_traces",
        "--mep_dir", str(target_mep_dir),
        "--out", str(trace_out_file)
    ]
    if not run_cmd(trace_cmd, project_dir, "Stage 3: Scoring Traces & Citations"):
        sys.exit(1)

    # D. Classify Errors (error_taxonomy)
    tax_out_file = project_dir / "output" / "error_taxonomy.jsonl"
    tax_cmd = [
        python_bin, "-m", "sme_capital_eval.eval.error_taxonomy",
        "--mep_dir", str(target_mep_dir),
        "--out", str(tax_out_file)
    ]
    if not run_cmd(tax_cmd, project_dir, "Stage 4: Compiling Error Taxonomy"):
        sys.exit(1)

    # E. Print and Save Summary (summarize)
    summary_out_file = project_dir / "output" / "summary.csv"
    summary_cmd = [
        python_bin, "-m", "sme_capital_eval.eval.summarize",
        "--metrics", str(eval_out_file),
        "--out", str(summary_out_file)
    ]
    if not run_cmd(summary_cmd, project_dir, "Stage 5: Compiling Consolidated Summary"):
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 UNIFIED EVALUATION PIPELINE RUN COMPLETE")
    print("=" * 60)
    print(f"Isolated MEPs Directory    : {target_mep_dir}")
    print(f"Scored Metrics JSONL       : output/metrics.jsonl")
    print(f"Trace Citations JSONL      : output/trace_metrics.jsonl")
    print(f"Error Classification JSONL : output/error_taxonomy.jsonl")
    print(f"Consolidated CSV Summary   : output/summary.csv")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
