# Causal mediation analysis of theory of mind in transformer language models

We patch activations between matched prompt pairs differing in one factor, per attention
head, to separate the heads that track **belief** from those doing **location retrieval**
and **state tracking**.

- primary model: Qwen2.5-14B-Instruct (48 layers × 40 heads = 1920 heads)
- replication: Gemma-2-27B-it (`--ungroup_grouped_query_attention`)
- templates: food_truck, hair_styling, library_book (+ a code-domain `indexed_list` arm)

## setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: conda env create -f cma_scripts/environment.yaml

export HF_TOKEN="..."                    # gated repos; https://huggingface.co/settings/tokens
export HF_HOME="/path/to/cache"          # optional
```


## quick start

```bash
cd cma_scripts/tasks/causal_analysis

python cma.py \
  --use_template_system \
  --context_type abstract \
  --base_rule ABA \
  --model_type Qwen2.5-14B-Instruct \
  --activation_name z \
  --prompt_num 50 \
  --sample_size 4 \
  --min_valid_sample_num 10 \
  --run_statistical_tests
```

`--use_template_system` (ToM) or `--context_type code` is required; omitting both raises.
`--patch_after_movement` selects the belief-formation patch position; omitting it patches
the end token. All headline results use the end token.

Statistics are a max-statistic permutation test with FWER correction over all 1920 heads,
5000 permutations in `tasks/causal_analysis/statistical_testing.py`. We recommend importing it rather than
reimplementing, so correction stays consistent across conditions.

## conditions

Canonical numbering lives in `analysis_scripts/condition_registry.py` (`REGISTRY`).

| #  | folder name                          | what varies (base↔exp)          | answers   | what it isolates                  |
|----|--------------------------------------|---------------------------------|-----------|-----------------------------------|
| C1 | `abstract_belief`                    | belief type (false↔true)        | same      | belief-divergence heads           |
| C2 | `abstract_photo`                     | photo timing (false↔true)       | same      | state-divergence heads (no beliefs) |
| C3 | `answer_changes_belief`              | belief type (false↔true)        | different | belief + answer heads (= C1 + C5) |
| C4 | `answer_changes_photo`               | photo timing (false↔true)       | different | state + answer heads              |
| C5 | `answer_changes_location_only`       | location tokens only, ToM frame | different | answer / location-retrieval heads |
| C6 | `answer_changes_location_only_photo` | location tokens only, photo     | different | answer / location-retrieval heads |
| C7 | `state_only_no_belief`               | state change, no belief         | different | state-update heads (control, no beliefs and no representations)      |

C1–C4 patch across contexts, so the rule flips (`ABA`↔`ABB`); C5–C7 patch within one
context and keep it. A parallel code domain uses copy-vs-alias list mutation as a non-ToM
analog (`CODE_REGISTRY` in the same file):

| #  | folder name                | mirrors | what it isolates |
|----|----------------------------|---------|------------------|
| K1 | `code_abstract`            | C1      | divergence heads, code frame (copy↔alias, answer same) |
| K3 | `code_answer_changes`      | C3      | divergence + answer (copy↔alias, answer differs) |
| K5 | `code_control`             | C5      | answer retrieval (letters swapped, copy/alias held) |
| Q1a–c | `code_index`, `code_index_flip`, `code_abstract_two_mut` | — | does a head track *which index* is queried? |
| Q2a–c | `code_veracity_only`, `code_site_only`, `code_site_only_rev` | — | copy status with the read site fixed, and the reverse |
| Q3a–b | `code_pos_matched`, `code_pos_unmatched` | — | answer-token position confound (pilot) |

The Q conditions use a held-out protocol: seed 0 selects heads, seed 1 tests them
(`analysis_scripts/analyze_code_index_dissociation.py`). Only `template_indexed_list`
runs count for the code domain.

## layout

| path | content |
|---|---|
| `cma_scripts/tasks/causal_analysis/` | the CMA pipeline: `cma.py`, `ablation_experiment.py`, `cma_config.py`, prompt generators |
| `cma_scripts/tom_datasets/` | prompt stimuli |
| `analysis_scripts/` | post-hoc analysis; also `circuit/`, `lookback_analysis/`, `rsa_head_outputs/`, `paper_figures/`, `tests/`. See its README, which also maps each paper figure to its script |
| `slurm_scripts/` | the SLURM launchers we ran, kept as the record of exact arguments (n, seeds, time limits). See its README |

Every module has a docstring with its own usage line; `condition_registry.py` is the
shared vocabulary most of them build on.

## results

```
results/causal_analysis/cma/{Qwen/Qwen2.5-14B-Instruct | google/gemma-2-27b-it_ungroup_gqa}/
  {condition}/template_{name}/base_rule_{base}_exp_rule_{exp}/
    {patch_position}/                    # patch_after_movement (bf) | patch_before_movement (end)
      z_seed_{S}_shuffle_False/logit/sample_num_{N}_gen_acc_0.9/
        causal_scores.pt, causal_scores_per_sample.pt, statistical_test_results.csv,
        significant_heads.csv, model_responses_*.txt
    z_seed_{S}_shuffle_False/{base,exp}_input_prompts_*.txt, ans_*.txt   # every generated prompt pair
```

Seed priority when resolving a source: `combined` > `1` > `0`. Use **seed=1, end position**
for the seven-condition analysis. `resid_post_*` folders are residual-stream arms.

`code_abstract` ships seed 0 (the runs behind the paper's code-domain cells) and seed 1
(added later as the held-out seed for the Q conditions); the seed priority picks seed 1, so
the code rows of `results/summary/h2_robustness/` are the seed-1 values. The paper quotes
only the story and Gemma rows, which do not depend on this choice.

Shipped alongside the CMA cells (derived tables only, no figures):

| path | content |
|---|---|
| `results/aggregated/` | fingerprints, cross-condition correlations and tests, head sets, head-selection inventory, path-patching and head-domain analyses |
| `results/ablation/` | cumulative and group ablation CSVs, random baselines |
| `results/circuit/` | Stage-1 configs and every Stage-2 steering run (`per_prompt.csv`, `summary.csv`, `attention.csv`) |
| `results/rsa_head_outputs/` | stimulus grids, RDM/Mantel tables, DLA, state geometry, per-head steering |
| `results/lookback_analysis/` | attention-mass tables per condition |
| `results/summary/` | the statistics behind the paper's tables and the heatmap manifest |
| `results/tom_performance/` | behavioral accuracy runs |

**Not shipped** (regenerable; multi-GB): activation captures (`results/circuit/stage1_capture/*.pt`,
`results/rsa_head_outputs/activations_*`), raw per-prompt path-patching outputs
(`results/circuit/path_patching/`, summarised in `results/aggregated/path_patching_*`),
and raw behavioral responses. The capture scripts are
`analysis_scripts/circuit/capture_circuit_activations.py`,
`analysis_scripts/rsa_head_outputs/capture_head_outputs*.py` and
`analysis_scripts/lookback_analysis/capture_attention_patterns.py`.

**Path:** CMA writes to `cma_scripts/results/` (its `log_dir` is relative and
`pyrootutils` chdirs into `cma_scripts/`), but every analysis script *reads*
`<repo_root>/results/`. Results must end up at the repository root.
