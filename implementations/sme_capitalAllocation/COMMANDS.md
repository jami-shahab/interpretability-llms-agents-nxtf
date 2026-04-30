# SME Capital Allocation — Copy-Paste Command Runbook

## Prerequisites (all machines)

```bash
# Clone the repo
git clone https://github.com/jami-shahab/interpretability-llms-agents-nxtf.git
cd interpretability-llms-agents-nxtf

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or restart terminal
```

---

## Environment Variables

Create `.env` in the repo root. Only include the key for the provider you are using:

```bash
# .env
GEMINI_API_KEY=your_gemini_api_key_here      # for --provider gemini
OPENAI_API_KEY=your_openai_api_key_here      # for --provider openai
# No key needed for --provider ollama
```

---

## Step 1 — Data Engineering (run once, all providers share the same data)

```bash
# Fast steps only (no HuggingFace required)
/home/shahab/.venv/bin/uv run \
  --env-file "$(pwd)/.env" \
  --directory implementations/sme_capitalAllocation \
  -m sme_capital_eval.data_engineering.run_all \
  --skip-finqa

# Or full pipeline including FinQA (requires internet + HuggingFace)
/home/shahab/.venv/bin/uv run \
  --env-file "$(pwd)/.env" \
  --directory implementations/sme_capitalAllocation \
  -m sme_capital_eval.data_engineering.run_all
```

---

## Step 2 — Generate MEPs

### OpenAI — 1 case (smoke test)

```bash
uv run \
  --env-file "$(pwd)/.env" \
  --directory implementations/sme_capitalAllocation \
  -m sme_capital_eval.runner.run_generate_meps \
  --provider openai \
  --model gpt-4o-mini \
  --split test \
  --n 1 \
  --workers 1 \
  --out meps/
```

### OpenAI — all 7 test cases

```bash
uv run \
  --env-file "$(pwd)/.env" \
  --directory implementations/sme_capitalAllocation \
  -m sme_capital_eval.runner.run_generate_meps \
  --provider openai \
  --model gpt-4o-mini \
  --split test \
  --n 7 \
  --workers 1 \
  --out meps/
```

### Gemini — all 7 test cases (free plan: run on a fresh UTC day)

```bash
uv run \
  --env-file "$(pwd)/.env" \
  --directory implementations/sme_capitalAllocation \
  -m sme_capital_eval.runner.run_generate_meps \
  --provider gemini \
  --model gemini-2.0-flash-lite \
  --split test \
  --n 7 \
  --workers 1 \
  --out meps/
```

### Ollama — all 7 test cases (local, no quota)

```bash
uv run \
  --directory implementations/sme_capitalAllocation \
  -m sme_capital_eval.runner.run_generate_meps \
  --provider ollama \
  --model gemma4:e2b \
  --ollama_url http://localhost:11434 \
  --split test \
  --n 7 \
  --workers 1 \
  --out meps/
```

---

## Step 3 — Evaluate

Replace `<config_name>` with the folder created under `meps/`:
- OpenAI gpt-4o-mini → `openai_gpt_4o_mini`
- Gemini flash-lite   → `gemini_gemini_2_0_flash_lite`
- Ollama gemma4:e2b  → `ollama_gemma4_e2b`

```bash
# Decision accuracy + per-lens + must-flag metrics
uv run \
  --directory implementations/sme_capitalAllocation \
  -m sme_capital_eval.eval.eval_outputs \
  --mep_dir meps/<config_name>/test \
  --out output/metrics.jsonl

# Latency + parse reliability
uv run \
  --directory implementations/sme_capitalAllocation \
  -m sme_capital_eval.eval.eval_traces \
  --mep_dir meps/<config_name>/test \
  --out output/trace_metrics.jsonl

# Error taxonomy
uv run \
  --directory implementations/sme_capitalAllocation \
  -m sme_capital_eval.eval.error_taxonomy \
  --mep_dir meps/<config_name>/test \
  --out output/error_taxonomy.jsonl

# Human-readable summary table
uv run \
  --directory implementations/sme_capitalAllocation \
  -m sme_capital_eval.eval.summarize \
  --metrics output/metrics.jsonl \
  --out output/summary.csv
```

---

## Quick Reference — Provider Defaults

| `--provider` | Default `--model` | Key required | Rate limits |
|---|---|---|---|
| `gemini` | `gemini-2.0-flash-lite` | `GEMINI_API_KEY` | 1500 req/day free tier |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` | Pay-per-token, no hard cap |
| `ollama` | `gemma4:e2b` | None | None (local) |
