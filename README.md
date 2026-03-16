# LLM Self-Correction Evaluation

**Before running:**

```bash
pip install -r requirements.txt
```

This project implements an evaluation harness for studying **self-correction strategies in large language models (LLMs)**.  
The goal is to measure **when iterative refinement improves model performance and when it simply increases computational cost**.

The system runs controlled experiments across multiple model families using standardized prompts, datasets, and evaluation metrics.

---

# Models Evaluated

The evaluation pipeline supports both **local open-weight models** and **API-based frontier models**.

### Local Models

- **Qwen 2.5 7B Instruct** (HuggingFace)
- **Llama 3 8B Instruct** (HuggingFace)

### API Models

- **Kimi K2 Instruct** (Fireworks API)
- **GPT-5.3 Chat** (OpenAI API)
- **Claude Sonnet** (Anthropic API)

These represent a mix of:

- open-weight transformer models  
- mixture-of-experts architectures  
- frontier proprietary models  

This diversity allows comparison of **how model scale and architecture affect self-correction behavior**.

---

# Datasets

The harness evaluates models on standardized reasoning benchmarks.

| Dataset | Task Type |
|--------|-----------|
| **GSM8K** | grade-school math reasoning |
| **HumanEval** | code generation with unit tests |
| **ARC** | multiple-choice scientific reasoning |

Using multiple datasets allows evaluation of whether self-correction improves performance **consistently across different problem types**.

---

# Self-Correction Strategies

The system currently implements several evaluation strategies.

| Strategy | Description |
|---------|-------------|
| **single_pass** | model produces one answer |
| **self_refine** | model generates → critiques → revises its answer |
| **best_of_n** | multiple samples generated and the best candidate selected |
| **oracle** | theoretical upper bound if the correct answer appears among candidates |

These strategies allow controlled comparison of **accuracy vs compute trade-offs**.

---

# Metrics Collected

For every model run the harness logs:

- model output  
- correctness vs gold answer  
- inference latency  
- token usage  

### Token Usage

- input tokens  
- output tokens  
- total tokens  

This enables analysis of:

- **accuracy**
- **latency**
- **token cost**
- **cost–performance tradeoffs**

Each experiment produces a structured log file:

```
runs/run_<timestamp>/results.jsonl
```

---

# Repository Structure

```
configs/       model and task configuration
data/          dataset loaders
models/        model provider implementations
strategies/    self-correction strategy implementations
evaluators/    scoring and evaluation logic
analysis/      scripts for computing metrics and plots
runs/          experiment outputs
harness.py     main experiment runner
logger.py      structured run logging
```

---

# Running the Harness

Example run:

```bash
python harness.py --model gpt --task gsm8k --strategy single_pass
```

Run a smaller test subset:

```bash
python harness.py --model llama --task gsm8k --strategy self_refine --limit 25
```

Results will be written to:

```
runs/run_<timestamp>/
```

Each run folder contains:

```
meta.json
results.jsonl
```

---

# Research Objective

This project investigates:

**When does iterative self-correction improve LLM reliability relative to its computational cost?**

The experiments measure:

- whether refinement strategies increase accuracy  
- how improvements vary across model scale  
- the **compute cost required to achieve those improvements**

The results help determine when self-correction is **beneficial, neutral, or wasteful**.