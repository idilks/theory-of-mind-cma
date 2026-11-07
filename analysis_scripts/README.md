# scripts

Analysis entry points are mapped in the root `CLAUDE.md` ("Analysis pipeline" table).
Experiment-to-script/output/report links are in [the evidence index](../docs/EXPERIMENTS.md);
dated open work remains in [STATUS.md](../docs/STATUS.md). Check existing artifacts
before proposing a repeat run, and leave uncertain provenance fields blank.
Import conditions from `condition_registry.py`; import stats from
`codebase/tasks/causal_analysis/statistical_testing.py`.

Scripts that write to `results_for_pi/...` are writing to `writeups/results_for_pi/`
(the root `results_for_pi` is a symlink, so HPC-side paths stay unchanged).

Legacy (old C-numbering, "C8" = C5; don't use for new results): `build_c8_comparison.py`,
`compare_bf_conditions.py`, `build_qmd.py`, `build_pi_qmd.py`, `build_pi_results.py`.
The previous README (pooled-analysis workflow with removed `context_type token`) is in
`archive/scripts_README_old.md`.
