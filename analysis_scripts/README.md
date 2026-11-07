# Analysis scripts

Post-hoc analysis over the CMA outputs. Every module has a docstring with a usage line;
the top-level README.md contains the condition table, the canonical registry and the
results layout. `condition_registry.py` contains the shared vocabulary: import conditions from it rather
than hand-rolling tables, and import the permutation test from
`cma_scripts/tasks/causal_analysis/statistical_testing.py`.

> We run everything here against a populated `results/` tree at the repository root. The
> derived data included in this repository is enough for the scripts marked **data-only**
> below; for the scripts marked **GPU** we load a model and regenerate the activation
> captures that are not included (see "not included" in the top-level README).

Paths are relative to the repository root; run scripts from there, e.g.
`python analysis_scripts/aggregate_conditions.py`.

## regenerating the paper's figures and tables

| paper figure | script | input data |
|---|---|---|
| Fig. 1, Fig. 2, Fig. A1 (TikZ schematics) | drawn by hand, not generated | — |
| condition heatmaps (`fig7_*`) | `paper_figures/fig7_fingerprint_heatmaps.py` | `results/aggregated/fingerprints/`, Gemma CMA cells |
| additive decomposition (`fig2b_*`) | `paper_figures/fig2b_h2_additivity_2panel.py` | `results/aggregated/`, `results/summary/h2_robustness/` |
| Gemma replication (`fig3_*`) | `paper_figures/fig3_gemma_replication.py` | Gemma CMA cells, `results/summary/` |
| steering attention (`steering_attention_paper`) | `circuit/render_steering_attention_simple.py` | `results/circuit/stage2_steering/` |
| code RDMs (`fig10_*`) | `paper_figures/fig10_code_rdms.py` | head-output captures (**not included**; we regenerate them with `rsa_head_outputs/capture_head_outputs_full.py`) |
| per-head axis steering (`fig11_*`) | `paper_figures/fig11_head_steering.py` | `results/rsa_head_outputs/steering_*` |
| story RDMs (`fig12_*`) | `paper_figures/fig12_story_rdms.py` | head-output captures (**not included**; same capture script) |
| steering coefficients (`fig13_*`) | `paper_figures/fig13_residual_coefficients.py` | `results/circuit/stage2_steering/` |
| coefficient extension (`fig14_*`) | `paper_figures/fig14_dose_extension.py` | `results/aggregated/circuit/dose_extension/` |
| content RDMs (`fig15_*`) | `paper_figures/fig15_content_rdms.py` | head-output captures (**not included**; same capture script on the content stimulus set) |
| appendix heatmap grids | `build_avg_heatmaps.py` (`--results-root` for Gemma) | CMA cells |

Figure output is in `results/figures/`. For the tables we use the CSVs produced with
`aggregate_conditions.py`, `controls.py`, `h2_additivity_robustness.py`,
`template_holdout_check.py`, `analyze_head_domain_profiles.py` and
`analyze_head_ablation_interactions.py`, all data-only.

## cross-condition aggregation (data-only)

| script | what we use it for |
|---|---|
| `condition_registry.py` | canonical registry (story C1–C7, code K/Q) and CMA result-path resolution; `python analysis_scripts/condition_registry.py` is a self-check of the registry against the results tree |
| `aggregate_conditions.py` | per-condition fingerprints, pairwise correlations, H1–H3 tests → `results/aggregated/` |
| `controls.py` | robustness controls for the aggregation (axis-matched substitutes, partial correlations, nulls) |
| `h2_additivity_robustness.py` | H2 additivity package: raw vs clamped, OLS, residuals, permutation nulls |
| `h2_gain_bootstrap.py` | bootstrap CI for the H2 additivity gain |
| `template_holdout_check.py` | leave-one-template-out check on the fingerprint correlations |
| `build_fingerprint_scatters.py`, `build_fingerprint_scatters_dual.py` | pairwise fingerprint scatters (clamped; clamped vs raw) |
| `analyze_pooled_conditions.py` | pooled permutation/FDR testing across templates (older workflow, usable on `causal_scores_per_sample.pt`) |
| `sweep_permutation_alpha.py`, `rerun_statistical_tests.py` | re-running or sweeping the FWER permutation test on saved per-sample scores |
| `combine_per_sample_runs.py`, `combine_aliased_c2_c3.py` | merging seed_0/seed_1 per-sample runs into a `combined` source |
| `head_profile_rsa.py`, `plot_rsa_mds.py` | head-level and cell-level RSA/MDS over the condition fingerprints |
| `aggregate_model_performance.py` | behavioral accuracy on the ToM stimuli across models (`results/tom_performance/`) |

## head sets and ablation

| script | what we use it for |
|---|---|
| `build_tom_c1_c5_head_csv.py`, `build_tom_c1_c5_full_ranking_csv.py` | C1/C5 head lists and the full 1,920-head ranking for cumulative ablation |
| `build_decomposition_head_csvs.py`, `build_intersection_head_csv.py` | belief-only / location-only / shared decompositions and the C1 ∩ C2 set |
| `build_code_head_csvs.py`, `build_gemma_head_sets.py` | code-domain and Gemma head sets from their CMA tests |
| `build_significance_head_sets.py` | significance-count selection as an alternative to fingerprint top-k |
| `random_baseline_ablation.py` | layer-matched random baseline for any head-set ablation (**GPU**) |
| `merge_cumulative_ablation_chunks.py` | merging chunked cumulative-ablation outputs |
| `compare_ablation_baselines.py`, `ablation_fb_vs_tb_permutation.py`, `plot_ablation_fb_tb_permutation.py` | ablation vs random null; paired sign-flip tests on FB−TB gaps |
| `analyze_head_domain_profiles.py`, `render_head_domain_comparison.py` | per-head CMA effects compared across domains from saved data (inputs in `results/aggregated/head_evidence_inventory/`) |
| `analyze_head_ablation_interactions.py` | belief/photo group-ablation interactions from saved runs |
| `render_ablation_figure.py`, `render_cross_frame_ablation.py`, `render_intersection_figure.py`, `render_c3_decomp_figure.py` | ablation and decomposition figures |

## heatmaps and reports

| script | what we use it for |
|---|---|
| `build_avg_heatmaps.py` | cross-template averaged heatmap + significance mask per condition (appendix grids) |
| `build_summary_heatmaps.py`, `make_singlecolor_heatmaps.py` | per-cell heatmaps and `results/summary/heatmaps/manifest.csv` |
| `build_code_domain_report.py` | figures and stats for the code-domain CMA report |
| `build_c8_comparison.py` | legacy bf-control comparison (old "C8" = C5 numbering) |

## prompt verification (data-only)

`verify_pair_generation.py`, `verify_cross_condition_pairs.py`, `verify_c6_prompts.py`,
`verify_code_index_pairs.py` (we execute each generated code snippet),
`test_patch_position_detection.py`, `diff_cma_vs_ablation_prompts.py`,
`audit_first_token_collisions.py`, `diagnose_library_book.py`, `expand_tom_vocabs.py`.

## code veracity-vs-address (Q conditions)

With `analyze_code_index_dissociation.py` we run the held-out test (heads selected on seed 0,
tested on seed 1) over `code_index`, `code_index_flip`, `code_abstract_two_mut`,
`code_veracity_only`, `code_site_only[_rev]` and the position pilot →
`results/aggregated/code_index_dissociation/`.

## circuit/ — residual-stream circuit, steering, path patching

In Stage 1 we run a geometric screen on captured activations; in Stage 2 we steer the
residual stream with the Stage-1 directions on held-out prompts.

| script | what we use it for |
|---|---|
| `build_circuit_head_sets.py` | writer and reader head sets for the circuit |
| `capture_circuit_activations.py` | Stage-1 capture (**GPU**) → `results/circuit/stage1_capture/` (not included) |
| `stage1_geometric_screen.py` | geometric screen on the capture → `results/aggregated/circuit/` |
| `stage2_steering.py` | Stage-2 steering with dose grid, controls, attention readout, collateral leg (**GPU**) |
| `stage2_shared_direction.py`, `extract_shared_direction.py`, `compare_wires.py` | shared ToM ∩ code direction and its steering |
| `stage2_reader_restoration.py`, `analyze_reader_restoration.py` | reader-head restoration under steering (**GPU** / summary) |
| `stage2_route_decomposition.py`, `analyze_route_decomposition.py` | three-route decomposition of the steering effect (**GPU** / summary) |
| `analyze_dose_extension.py` | coefficient extension to α = 16 with belief-free collateral readouts |
| `analyze_steering_attention_null.py`, `render_steering_attention_bars.py`, `render_steering_attention_simple.py` | reader-head attention under steering: nulls and figures |
| `check_steering_position.py`, `correct_projection_comparisons.py` | offline position check; Holm correction of the projection comparisons |
| `generic_prompts.py`, `steering_bridge.py`, `circuit_common.py` | shared prompts, head→residual bridge, constants |
| `path_patching.py`, `path_patching_study.py` | direct sender→receiver path patching with frozen writer heads: 41 senders plus layer-matched controls, one SLURM task per cell (**GPU**); analyses in `results/aggregated/path_patching_*` |
| `path_patching_clean_control.py`, `path_patching_h200_repair.py` | clean-transplant control, and re-running missing cells from the frozen snapshot |
| `test_*.py` | CPU tests on tiny transformers and launcher simulations; no model downloads |

## lookback_analysis/ — attention

| script | what we use it for |
|---|---|
| `capture_attention_patterns.py`, `capture_code_attention_rows.py` | attention captures for CMA-significant heads (**GPU**) |
| `per_condition_top_heads.py`, `head_groups.py` | head selection per condition; fixed groups for the region figures |
| `analyze_lookback_attention.py`, `cma_weighted_attention.py`, `cross_condition_heatmap.py`, `cross_condition_per_token.py`, `per_token_profile.py` | attention-mass summaries and figures |
| `region_attention_shares.py`, `code_region_shares.py`, `region_bubble_matrix.py` | final-token attention by prompt region, story and code |
| `prompt_attention_bars.py`, `build_attention_position_bars.py`, `aggregate_prompt_anatomy.py` | token-level bars and prompt anatomy |
| `render_head_attention_simple.py`, `render_move_roles.py` | summary figures |
| `diagnose_named_coverage.py`, `test_photo_detector.py`, `run_local_4bit.sh` | diagnostics; local 4-bit capture |

## rsa_head_outputs/ — RSA, DLA, geometry, steering by head

| script | what we use it for |
|---|---|
| `build_stimulus_set.py`, `build_content_stimulus_set.py`, `build_code_position_stimuli.py`, `build_code_ledger_stimuli.py`, `build_code_padtwins_stimuli.py` | balanced stimulus grids (story 2×2, content-separable, code position / ledger / pad-twins) |
| `capture_head_outputs.py`, `capture_head_outputs_full.py` | per-head output captures at the final token (**GPU**; not included) |
| `select_topk_heads.py`, `head_sets_source.py`, `dims.py`, `chat_compat.py`, `rsa_utils.py` | head selection and shared helpers (model-parameterised) |
| `analyze_rdms*.py`, `analyze_intersection_rdm.py`, `per_head_analysis.py`, `dissociation_plot.py`, `all_conditions_dissociation.py` | Mantel tests of head-output RDMs against belief-veracity and answer models |
| `analyze_content_rdms.py`, `mvpa_veracity.py` | content-separable RSA and linear decoding of veracity |
| `analyze_code_position_rdms.py`, `analyze_code_ledger_rdms.py`, `analyze_code_padtwins_rdms.py`, `analyze_midpad_targeting.py`, `diagnose_code_ledger_rdm.py`, `partial_significance_code_ledger.py`, `code_ledger_random_null.py`, `plot_code_ledger_random_null.py`, `null_floor_864.py` | code-domain RDM experiments and their nulls |
| `padtwins_projection_test.py`, `padtwins_position_prediction.py`, `padtwins_causal_ablation.py`, `padtwins_followups.py` | pad-twins deconfound (projection, position prediction, causal ablation) |
| `compute_dla.py`, `plot_dla.py` | direct logit attribution per head |
| `state_geometry_alignment.py`, `state_geometry_threeway.py` | belief / photo / code state-axis geometry |
| `steer_state_axes.py`, `steer_photo_axis_on_code.py`, `cross_patch_experiment.py`, `patch_common.py` | per-head axis steering and cross-patching (**GPU**) |
| `code_idx_pairs.py`, `code_addressing_pairs.py` | code prompt pairs for the index and addressing contrasts |
| `plot_code_rdm_figures.py` | code RDM figures |

## paper_figures/

`fig*.py` are the figure scripts listed above; `style.py` contains the shared palette and
rcParams; `slides_simple_scatters.py` is a presentation variant.

## tests/

We run the small CPU checks for the head-domain and ablation-interaction analyses with
`python -m unittest discover -s analysis_scripts/tests`.
