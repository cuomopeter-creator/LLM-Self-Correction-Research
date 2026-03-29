# Project Finish Checklist

This checklist maps the remaining work needed to finish the research project based on:

- `/mnt/f/milestone_5_draft.docx`
- `/mnt/f/milestone_5_requirements.docx`
- `/mnt/f/Pre-final Feedback.docx`
- the current project state in this repo

## 1. Lock The Final Experiment Set

- [x] Finish the remaining reruns and active benchmark sweeps.
- [x] Decide which runs are included in the final paper.
- [x] Update `analysis/run_manifest.csv` to reflect the authoritative final run set.
- [x] Mark incomplete or superseded runs clearly so they are not accidentally used in the paper.

## 2. Rebuild The Final Analysis Layer

- [x] Recompute the final master metrics from the final included runs.
- [x] Restore or implement `analysis/instance_taxonomy.py`.
- [x] Generate instance-level transition counts for:
- [x] `correct -> correct`
- [x] `incorrect -> correct`
- [x] `correct -> incorrect`
- [x] `incorrect -> incorrect`
- [x] Compute revision success rate by model/task/strategy.
- [x] Compute error amplification rate by model/task/strategy.
- [x] Compute tokens per correct answer.
- [x] Compute accuracy per 1,000 tokens.
- [x] Add bootstrap confidence intervals or other simple statistical support for key strategy comparisons.

## 3. Upgrade The Main Results Presentation

- [x] Create a main accuracy table for `model x task x strategy`.
- [x] Create a compute-efficiency table.
- [x] Create a transition/regression summary table.
- [x] Create a heatmap showing delta accuracy relative to single-pass.
- [x] Create one compute-normalized figure such as tokens per correct answer or accuracy per 1,000 tokens.
- [x] Create a master summary figure for the main results narrative.
- [ ] Move overflow detail to appendix material instead of relying on many embedded charts in the main body.

## 4. Strengthen The Paper Itself

- [x] Tighten the central claim: self-correction is conditional, not universally beneficial.
- [x] State explicit research questions or hypotheses in the introduction.
- [x] Expand related work using the references already gathered in `references/references.txt`.
- [x] Ensure the paper has a complete Methodology / System Design section, not just scattered implementation description.
- [x] Ensure the paper has a clear Data & Experimental Setup section covering datasets, preprocessing, assumptions, and evaluation protocol.
- [x] Ensure the paper has a clear Implementation Details section covering software architecture, tools, hardware/compute resources, and reproducibility notes.
- [ ] Revise the results section to focus on the strongest findings supported by the final tables.
- [x] Expand the discussion section with:
- [x] where self-refine helps
- [x] where self-refine hurts
- [x] where self-refine wastes compute
- [x] where best-of-N is the safer choice
- [x] where oracle exposes latent capability gaps
- [x] Add clear limitations and failure-case discussion.
- [x] Decide whether to include the simple predictive decision framework suggested in feedback, or explicitly omit it from the final scope.
- [x] Add a Conclusion and Future Work section.
- [ ] Make sure every figure and table is numbered, captioned, and referenced in the text.
- [x] Make sure the references section is complete and consistently formatted.

## Completed Writing So Far

- [x] Abstract completed.
- [x] Introduction completed.

## 5. Reproducibility And Submission Readiness

- [x] Rewrite `README.md` into a real end-to-end reproducibility guide.
- [x] Document environment setup, dependencies, and Python version.
- [x] Document operating system assumptions and any recommended development environment or IDE expectations.
- [x] Document model/API setup and expected environment variables.
- [x] Document dataset setup and any download/preprocessing steps.
- [x] Explain the project structure and folder layout clearly.
- [x] Document exact commands to rerun experiments.
- [x] Document exact commands to regenerate tables and figures.
- [x] Add approximate runtime expectations.
- [ ] State whether datasets are included directly or must be downloaded, and document the exact source plus file-layout expectations.
- [ ] Ensure the code package can be followed without hidden assumptions.
- [ ] Verify the submitted code package is runnable, readable, and adequately commented for an instructor walkthrough.

## 6. Code Walkthrough / Recording Deliverable

- [ ] Prepare a clean walkthrough path through the repo structure.
- [ ] Record a screen walkthrough of the major files and execution flow.
- [ ] Demonstrate successful execution of the harness and analysis pipeline.
- [ ] Show how results are produced and where outputs are written.

## 7. Final Packaging

- [ ] Clean the submission state of the repo.
- [ ] Make a deliberate choice about which large artifacts are submitted versus referenced.
- [x] Verify the final `.docx` includes all required Milestone 5 sections.
- [ ] Export the final paper as `.docx`.
- [x] Verify the first-page metadata required by Milestone 5 is present.
- [ ] Verify the file naming matches the course requirement exactly.
- [ ] Verify the final word count target is met.
- [ ] Verify citations and references are complete and consistent.

## Recommended Execution Order

1. Finish runs and lock the final manifest.
2. Rebuild metrics and transition analysis.
3. Generate final tables and figures.
4. Rewrite results/discussion around those outputs.
5. Rewrite the README for reproducibility.
6. Record the walkthrough video.
7. Final package and submission pass.

## Current Project Read

- Experimental execution is already strong.
- The biggest remaining gaps are analysis integration, final paper polish, and reproducibility packaging.
- The project is much closer to done on experimentation than it is on writeup/submission polish.
