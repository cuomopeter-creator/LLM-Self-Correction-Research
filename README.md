# LLM Self-Correction Evaluation

This project implements an evaluation harness for studying **self-correction strategies in large language models (LLMs)**.
The goal is to measure when iterative refinement improves results and when it simply increases compute cost.

The system runs controlled experiments across multiple model families using standardized prompts, datasets, and evaluation metrics.

## Models Used

The evaluation pipeline currently supports:

* **Qwen 2.5 7B Instruct** (local HuggingFace)
* **Llama 3 8B Instruct** (local HuggingFace)
* **Kimi K2 Instruct** (MoE model via Fireworks API)
* **GPT-5.3 Chat** (OpenAI API)
* **Claude Sonnet** (Anthropic API)

These represent a mix of:

* open-weight models
* mixture-of-experts architectures
* frontier proprietary models

## Datasets

Experiments will be conducted across multiple benchmarks to test reasoning and factual reliability:

* **GSM8K** – grade school math reasoning problems
* **StrategyQA** – multi-hop reasoning with implicit knowledge
* **HotpotQA** – multi-hop question answering requiring reasoning across multiple facts

Using multiple datasets allows evaluation of whether self-correction improves performance **consistently across task types**.

## Metrics Collected

For every model run the harness logs:

* model output
* correctness vs gold answer
* inference latency
* **token usage**

  * input tokens
  * output tokens
  * total tokens

This enables analysis of **accuracy vs computational cost**.

## Repository Structure

```
configs/       model and task configs
data/          dataset loaders
models/        model provider implementations
strategies/    self-correction strategies
evaluators/    scoring and evaluation scripts
analysis/      result analysis and plotting
runs/          experiment outputs
harness.py     main experiment runner
logger.py      structured run logging
```

## Running the Harness

Example:

```
python harness.py --model gpt --limit 25
```

This runs the selected model on a subset of the evaluation dataset and logs results for analysis.

## Research Goal

The project investigates:

**When does iterative self-correction improve LLM reliability relative to its computational cost?**
