# README For Codex

This file is a handoff note for the next Codex instance working in this repo.

## Immediate Context

- Repo root: `/mnt/f/self_correction_eval`
- Main status tracker: [`PROJECT_FINISH_CHECKLIST.md`](/mnt/f/self_correction_eval/PROJECT_FINISH_CHECKLIST.md)
- The user is actively polishing the paper and analysis outputs, not building new experiment infrastructure.
- The user explicitly does **not** want orchestration shell scripts for running experiments.
- The user interacts with the harness directly using commands like:

```bash
python harness.py --model <model> --task <task> --strategy <strategy> --limit <n>
```

## Important Scope Notes

- DeepSeek is **not** part of the paper’s experiment scope.
- Do not assume newer exploratory runs are part of the final paper.
- The user said they know what the final experiment is; do not re-argue the experimental definition unless directly asked.
- Keep the focus on paper completion, analysis presentation, README/reproducibility, and packaging.

## What Was Done In This Session

### Checklist updates

The following items were marked done in [`PROJECT_FINISH_CHECKLIST.md`](/mnt/f/self_correction_eval/PROJECT_FINISH_CHECKLIST.md):

- final run selection
- superseded/incomplete run handling
- main accuracy table
- compute-efficiency table
- transition/regression summary table
- delta-accuracy heatmap
- compute-normalized figure
- master summary figure

### Master summary figure

A new master tradeoff figure was added and iterated several times.

Current output:

- [`analysis/figures/master_strategy_tradeoff_scatter.html`](/mnt/f/self_correction_eval/analysis/figures/master_strategy_tradeoff_scatter.html)

Current implementation:

- [`analysis/plot_results.py`](/mnt/f/self_correction_eval/analysis/plot_results.py)

Current figure design:

- faceted by model
- color = strategy
- shape = task
- model shown in hover
- interior axis labels removed
- outside axis labels retained
- x-axis label also shown on the Llama panel by user request
- quadrant shading added
- Pareto frontier added
- quadrant descriptors are bold, larger, and color-coded

The user liked this version and said it can come off the to-do list.

## Existing Analysis Outputs

### Manifest / master tables

- [`analysis/run_manifest.csv`](/mnt/f/self_correction_eval/analysis/run_manifest.csv)
- [`analysis/master_results.csv`](/mnt/f/self_correction_eval/analysis/master_results.csv)

### Derived data tables

- [`data/compute_efficiency.csv`](/mnt/f/self_correction_eval/data/compute_efficiency.csv)
- [`data/bootstrap_accuracy_cis.csv`](/mnt/f/self_correction_eval/data/bootstrap_accuracy_cis.csv)
- [`data/bootstrap_accuracy_ci_samples.csv`](/mnt/f/self_correction_eval/data/bootstrap_accuracy_ci_samples.csv)
- [`data/instance_taxonomy_summary_single_pass_vs_best_of_n.csv`](/mnt/f/self_correction_eval/data/instance_taxonomy_summary_single_pass_vs_best_of_n.csv)
- [`data/instance_taxonomy_summary_single_pass_vs_self_refine.csv`](/mnt/f/self_correction_eval/data/instance_taxonomy_summary_single_pass_vs_self_refine.csv)
- [`data/instance_taxonomy_summary_self_refine_draft_vs_final.csv`](/mnt/f/self_correction_eval/data/instance_taxonomy_summary_self_refine_draft_vs_final.csv)

### Plot outputs

Directory:

- [`analysis/figures`](/mnt/f/self_correction_eval/analysis/figures)

Important current figures:

- model-level accuracy scatter HTMLs
- model-level token usage scatter HTMLs
- [`analysis/figures/efficiency_delta_heatmap.html`](/mnt/f/self_correction_eval/analysis/figures/efficiency_delta_heatmap.html)
- [`analysis/figures/bootstrap_ci_model_distributions.html`](/mnt/f/self_correction_eval/analysis/figures/bootstrap_ci_model_distributions.html)
- [`analysis/figures/master_strategy_tradeoff_scatter.html`](/mnt/f/self_correction_eval/analysis/figures/master_strategy_tradeoff_scatter.html)

## What Still Appears Left

Based on the current checklist and repo state, the remaining work is now mostly:

### Main paper / writing

- move overflow detail to appendix instead of relying on too many charts in the main body
- tighten the central claim
- expand related work
- revise results section around strongest findings
- expand discussion:
  - where self-refine helps
  - where self-refine hurts
  - where self-refine wastes compute
  - where best-of-N is safer
  - where oracle exposes latent capability
- add limitations and failure cases
- make sure every figure/table is numbered, captioned, and cited in text

### Reproducibility

- rewrite [`README.md`](/mnt/f/self_correction_eval/README.md) into a real end-to-end reproducibility guide
- document environment setup, Python version, dependencies
- document model/API setup and env vars
- document dataset setup and preprocessing
- document exact commands for experiments
- document exact commands to regenerate tables/figures
- add runtime expectations
- remove hidden assumptions

### Walkthrough / submission

- prepare repo walkthrough
- record walkthrough/demo
- demonstrate harness + analysis flow
- clean submission state
- decide which large artifacts to submit vs reference
- export final `.docx`
- verify metadata, naming, word count, references

## Current Top-Level Files

Useful top-level files in repo root:

- [`harness.py`](/mnt/f/self_correction_eval/harness.py)
- [`logger.py`](/mnt/f/self_correction_eval/logger.py)
- [`loaders.py`](/mnt/f/self_correction_eval/loaders.py)
- [`README.md`](/mnt/f/self_correction_eval/README.md)
- [`PROJECT_FINISH_CHECKLIST.md`](/mnt/f/self_correction_eval/PROJECT_FINISH_CHECKLIST.md)
- [`requirements.txt`](/mnt/f/self_correction_eval/requirements.txt)
- [`requirements-lock.txt`](/mnt/f/self_correction_eval/requirements-lock.txt)
- [`requirements-lock.no-torch.txt`](/mnt/f/self_correction_eval/requirements-lock.no-torch.txt)

## Parent Folder Reference Docs

The repo checklist was originally informed by these parent-folder docs:

- `/mnt/f/milestone_5_draft.docx`
- `/mnt/f/milestone_5_requirements.docx`
- `/mnt/f/Pre-final Feedback.docx`

If the next Codex instance needs context on assignment requirements or paper framing, those are the relevant docs.

## User Preferences / Interaction Notes

- Keep responses concise and practical.
- Do not overcomplicate experiment execution.
- Do not create orchestration shell scripts unless the user explicitly asks.
- Prefer direct edits and direct `harness.py` usage.
- When updating progress, be concrete about what changed and which file contains it.
- The user is comfortable steering analysis/paper decisions interactively.

## Suggested Next Best Actions

If picking up from here, the highest-value next tasks are probably:

1. Rewrite [`README.md`](/mnt/f/self_correction_eval/README.md) as a real reproducibility guide.
2. Help convert existing CSV/HTML outputs into paper-ready tables/captions.
3. Help draft/revise the results and discussion sections around the already-generated analysis.
4. Help with final repo cleanup and submission packaging.
