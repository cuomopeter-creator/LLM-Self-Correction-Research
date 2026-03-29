# LLM Self-Correction Evaluation

This repository contains the evaluation harness, run logs, analysis scripts, and paper artifacts for a controlled study of when inference-time self-correction helps, hurts, or wastes compute in large language models.

The project compares four strategies:

- `single_pass`
- `best_of_n`
- `self_refine`
- `oracle`

across multiple tasks and model families while tracking:

- accuracy
- token usage
- latency
- transition behavior such as `correct -> incorrect` regressions

## Operating Assumptions

- OS: Linux is the primary tested environment.
- Python: use Python `3.11+`.
- GPU: required for local Hugging Face models such as `llama` and `qwen`.
- Network access: required for API-based models and for any dataset/model downloads not already cached locally.

## Environment Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want a more reproducible pinned install, use one of:

```bash
pip install -r requirements-lock.txt
```

or, if you do not want to install Torch from the lockfile:

```bash
pip install -r requirements-lock.no-torch.txt
```

## Required Environment Variables

The harness loads environment variables from `.env` via `python-dotenv`, so the easiest setup is to create a repo-local `.env` file.

API-backed models require:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
FIREWORKS_API_KEY=...
```

Notes:

- `gpt` uses `OPENAI_API_KEY`
- `claude` uses `ANTHROPIC_API_KEY`
- `kimi` uses `FIREWORKS_API_KEY`
- local Hugging Face models do not require API keys, but they do require working local model access and GPU memory

## Model Configuration

The main model definitions live in [configs/models.yaml](/mnt/f/self_correction_eval/configs/models.yaml).

The current paper-facing model keys are:

- `qwen`
- `llama`
- `kimi`
- `gpt`
- `claude`

Additional exploratory entries may exist in the config file but are not necessarily part of the final paper.

## Project Structure

Key directories and files:

```text
self_correction_eval/
├── analysis/        analysis scripts and aggregated outputs
├── configs/         model configuration
├── data/            cached benchmark data plus derived CSV outputs
├── evaluators/      task-specific scoring logic
├── models/          provider/model wrappers
├── references/      local paper/reference PDFs and reference list
├── runs/            per-run outputs written by the harness
├── strategies/      inference strategies
├── harness.py       main experiment runner
├── logger.py        structured JSONL logging
├── dashboard.py     lightweight results viewer
└── README.md
```

Important generated artifacts:

- [analysis/run_manifest.csv](/mnt/f/self_correction_eval/analysis/run_manifest.csv)
- [analysis/master_results.csv](/mnt/f/self_correction_eval/analysis/master_results.csv)
- [data/compute_efficiency.csv](/mnt/f/self_correction_eval/data/compute_efficiency.csv)
- [data/bootstrap_accuracy_cis.csv](/mnt/f/self_correction_eval/data/bootstrap_accuracy_cis.csv)
- [data/instance_taxonomy_summary_single_pass_vs_self_refine.csv](/mnt/f/self_correction_eval/data/instance_taxonomy_summary_single_pass_vs_self_refine.csv)
- [analysis/figures/](/mnt/f/self_correction_eval/analysis/figures)

## Datasets

This project uses:

- `gsm8k`
- `truthfulqa`
- `humaneval`
- `arc`

The harness loads datasets through [data/loaders.py](/mnt/f/self_correction_eval/data/loaders.py). In practice, that means:

- some benchmark files may already be cached under [data/](/mnt/f/self_correction_eval/data)
- additional dataset downloads may occur automatically depending on your local cache state

If you are reproducing from scratch, expect the first run to download missing benchmark artifacts.

## Running Experiments

The main entrypoint is [harness.py](/mnt/f/self_correction_eval/harness.py).

Example single run:

```bash
python harness.py --model gpt --task gsm8k --strategy single_pass
```

Run a smaller subset:

```bash
python harness.py --model llama --task gsm8k --strategy self_refine --limit 25
```

Run all tasks for one strategy:

```bash
python harness.py --model claude --task all --strategy best_of_n
```

Run all strategies for one task:

```bash
python harness.py --model kimi --task truthfulqa --strategy all
```

Outputs are written to:

```text
runs/run_<timestamp>/
```

Each run directory contains at least:

```text
meta.json
results.jsonl
```

## Supported Tasks and Strategies

`harness.py` currently supports:

Tasks:

- `gsm8k`
- `truthfulqa`
- `humaneval`
- `arc`

Strategies:

- `single_pass`
- `best_of_n`
- `self_refine`
- `oracle`

Notes:

- `humaneval` is automatically capped at `40` examples by the harness
- `oracle` is skipped for `humaneval`

## Regenerating Analysis Outputs

The final paper analysis is driven by the run manifest in [analysis/run_manifest.csv](/mnt/f/self_correction_eval/analysis/run_manifest.csv).

### 1. Recompute aggregate metrics

```bash
python analysis/compute_metrics.py --manifest analysis/run_manifest.csv --output analysis/master_results.csv
```

### 2. Recompute compute-efficiency summaries

```bash
python analysis/compute_efficiency.py --manifest analysis/run_manifest.csv --output data/compute_efficiency.csv
```

### 3. Recompute bootstrap confidence intervals

```bash
python analysis/bootstrap_cis.py --manifest analysis/run_manifest.csv --output data/bootstrap_accuracy_cis.csv --samples-output data/bootstrap_accuracy_ci_samples.csv
```

### 4. Recompute instance-taxonomy summaries

```bash
python analysis/instance_taxonomy.py --manifest analysis/run_manifest.csv --output-dir data
```

### 5. Regenerate figures

```bash
python analysis/plot_results.py --input analysis/master_results.csv --efficiency-input data/compute_efficiency.csv --output-dir analysis/figures
```

Optional bootstrap plotting:

```bash
python analysis/plot_bootstrap_cis.py --input data/bootstrap_accuracy_cis.csv --samples-input data/bootstrap_accuracy_ci_samples.csv --output-dir analysis/figures
```

## Expected Runtime

Runtime depends heavily on model choice:

- local open-weight models: slower setup, GPU-dependent, no API costs
- API models: faster to start, but subject to network/API latency and usage costs

Rough expectations:

- small smoke test with `--limit 25`: minutes
- one full `200`-example API run: tens of minutes to longer, depending on provider
- recomputing CSV analysis outputs from existing runs: usually minutes
- figure generation from existing CSVs: usually under a few minutes

The most expensive part of reproduction is re-running the model evaluations, not rebuilding the analysis CSVs or plots.

## Reproducibility Notes

- The harness logs every example to `results.jsonl`, which is the main audit trail for downstream analysis.
- The final paper should use the run set listed in [analysis/run_manifest.csv](/mnt/f/self_correction_eval/analysis/run_manifest.csv), not arbitrary newer exploratory runs.
- Token usage is the main compute-efficiency metric used in the paper.
- Latency is logged, but it is not treated as the primary efficiency measure because results mix local GPU execution and multiple external APIs.
- Some stochastic variation is expected for generative model outputs unless providers and local inference paths are fully pinned.

## Dashboard / Inspection

There is a lightweight dashboard entrypoint:

```bash
python dashboard.py
```

There is also an exploratory notebook at:

- [analysis/explore_analysis.ipynb](/mnt/f/self_correction_eval/analysis/explore_analysis.ipynb)

## Code Walkthrough Path

For a walkthrough or demo, the cleanest path is:

1. [harness.py](/mnt/f/self_correction_eval/harness.py)
2. [models/](/mnt/f/self_correction_eval/models)
3. [strategies/](/mnt/f/self_correction_eval/strategies)
4. [evaluators/](/mnt/f/self_correction_eval/evaluators)
5. one example run directory under [runs/](/mnt/f/self_correction_eval/runs)
6. [analysis/compute_metrics.py](/mnt/f/self_correction_eval/analysis/compute_metrics.py)
7. [analysis/compute_efficiency.py](/mnt/f/self_correction_eval/analysis/compute_efficiency.py)
8. [analysis/instance_taxonomy.py](/mnt/f/self_correction_eval/analysis/instance_taxonomy.py)
9. [analysis/plot_results.py](/mnt/f/self_correction_eval/analysis/plot_results.py)

## Final Paper Scope Notes

- The final paper scope excludes `deepseek`.
- Do not assume every run in [runs/](/mnt/f/self_correction_eval/runs) is part of the final manuscript.
- The authoritative final analysis should follow the curated manifest and the generated aggregate outputs listed above.
