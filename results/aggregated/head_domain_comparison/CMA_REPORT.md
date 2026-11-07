# Existing-data head comparison

This analysis ran on saved tensors; no language-model inference was performed.

Metric: CMA is the patched minus clean target-versus-original answer logit margin. Zero means no change, positive means movement toward the specified counterfactual answer, and the range is unbounded. It is not accuracy, probability, or a head-ablation effect. Cell means are equally weighted in summaries; templates and directions remain separately available.

Uncertainty: joint prompt-row bootstrap within each cell, calibrated using the maximum standardized mean error across all 1,920 heads. Per-cell error budgets sum to 0.05 across all source cells. Propagated bands therefore target simultaneous 95% coverage across the displayed linear contrasts under the bootstrap approximation, without assuming independence or pairing across cells. This is conservative; historical head selection is not an independent replication. Zero-variance uncertainties are left blank.

All cells are conditioned on both donor and recipient passing original accuracy filters. Cross-domain differences are differences in intervention effects on these filtered prompt populations, not an isolated causal effect of domain. Code has only one template.
## Strictly belief-only heads

Each entry is mean difference [simultaneous lower, upper], in raw logit-margin units. Positive belief−photo/code means a larger measured belief patch effect; a bound covering zero is unresolved, not evidence of equivalence. These averages are conditional on the observed templates.

| Head | Belief − photo | Belief − code |
|---|---|---|
| L29H11 | -0.007 [-0.419, 0.406] | -0.029 [-0.502, 0.444] |
| L31H17 | 1.032 [0.619, 1.446] | 0.917 [0.417, 1.417] |
| L31H21 | 0.422 [0.154, 0.690] | 0.509 [0.241, 0.777] |
| L32H26 | 0.444 [0.060, 0.829] | 0.412 [0.033, 0.790] |
| L33H1 | 0.252 [0.011, 0.494] | 0.266 [0.042, 0.491] |
| L36H10 | -0.042 [-0.569, 0.485] | -0.069 [-0.647, 0.510] |
| L36H12 | -0.026 [-0.521, 0.468] | -0.180 [-0.772, 0.411] |
| L36H24 | 0.191 [-0.296, 0.678] | -0.178 [-0.626, 0.269] |
| L36H33 | 0.138 [-0.178, 0.453] | -0.272 [-0.654, 0.111] |
| L37H15 | -0.075 [-0.409, 0.260] | -0.188 [-0.520, 0.144] |
| L46H10 | -0.144 [-0.546, 0.258] | -0.271 [-0.809, 0.267] |
| L46H13 | 0.212 [-0.114, 0.539] | 0.308 [-0.127, 0.743] |

## Files

- `cma_cell_profiles.csv`: every selected head, condition/template/direction, both pointwise 95% bootstrap intervals and family-wide cell bands. Do not confuse their multiplicity scopes.
- `cma_head_comparisons.csv`: all planned cell and aggregate differences, exact weights, uncertainty, seed-1 sensitivity.
- `cma_all_heads_aggregate_comparisons.csv`: aggregate contrasts for all 1,920 heads, before selection.
- `cma_source_manifest.json` and `cma_analysis_config.json`: hashes, shapes, seeds, method and source revision.
- `cma_profiles.png`: exploratory raw-score profile, not a significance map.

Controls answer different intervention questions: C1−C5 and the Q contrasts do not subtract out a uniquely identified semantic component. They constrain hypotheses but do not establish what a head exclusively encodes.

Seed-1 sensitivity missing sources: 0. No matching row IDs are assumed across source cells.
