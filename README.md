# Causal mediation analysis of theory of mind in transformer language models

We patch activations between matched prompt pairs differing in one factor, per attention
head, to separate heads that encode abstract divergence information from heads responsible
for retrieval.

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

One of `--use_template_system` (ToM) or `--context_type code` is required.

For statistics we use a max-statistic permutation test with FWER correction over all 1920
heads, 5000 permutations, implemented in `tasks/causal_analysis/statistical_testing.py`. We
recommend importing for consistency across conditions.

## conditions

Each condition is named by what changes between the two prompts (divergence, answer, or both)
and by its domain. The canonical table is `REGISTRY` in `analysis_scripts/condition_registry.py`.

| #  | paper name               | folder name                          | changes between prompts       | answers   | target heads |
|----|--------------------------|--------------------------------------|-------------------------------|-----------|--------------|
| C1 | divergence-change belief | `abstract_belief`                    | false ↔ true belief           | same      | divergence heads |
| C2 | divergence-change photo  | `abstract_photo`                     | outdated ↔ current photo      | same      | divergence heads, no agent |
| C3 | dual-change belief       | `answer_changes_belief`              | false ↔ true belief           | different | divergence + retrieval (= C1 + C5) |
| C4 | dual-change photo        | `answer_changes_photo`               | outdated ↔ current photo      | different | divergence + retrieval |
| C5 | answer-change belief     | `answer_changes_location_only`       | location tokens, belief frame | different | retrieval heads |
| C6 | answer-change photo      | `answer_changes_location_only_photo` | location tokens, photo frame  | different | retrieval heads |
| C7 | state-only control       | `state_only_no_belief`               | location tokens, no agent     | different | retrieval heads, no divergence |

In C1–C4 we patch across contexts, so the rule flips (`ABA`↔`ABB`); in C5–C7 we patch within
one context. The code domain mirrors the story contrasts with copy-vs-alias list mutation
(`CODE_REGISTRY` in the same file):

| #  | paper name              | folder name            | story analog | changes between prompts |
|----|-------------------------|------------------------|--------------|-------------------------|
| K1 | divergence-change code  | `code_abstract`        | C1 | copy ↔ alias, answer same |
| K3 | dual-change code        | `code_answer_changes`  | C3 | copy ↔ alias, answer differs |
| K5 | answer-change code      | `code_control`         | C5 | letters swapped, copy/alias held |
| Q1a–c | — | `code_index`, `code_index_flip`, `code_abstract_two_mut` | — | queried index |
| Q2a–c | — | `code_veracity_only`, `code_site_only`, `code_site_only_rev` | — | copy status with the read site fixed, and the reverse |
| Q3a–b | — | `code_pos_matched`, `code_pos_unmatched` | — | answer-token position (pilot) |

The Q conditions test whether divergence heads track veracity or the queried address. We
select heads on seed 0 and test them on seed 1 (`analysis_scripts/analyze_code_index_dissociation.py`).
For the code domain we count only `template_indexed_list` runs.

## layout

| path | content |
|---|---|
| `cma_scripts/tasks/causal_analysis/` | the CMA pipeline: `cma.py`, `ablation_experiment.py`, `cma_config.py`, prompt generators |
| `cma_scripts/tom_datasets/` | prompt stimuli |
| `analysis_scripts/` | post-hoc analysis; also `circuit/`, `lookback_analysis/`, `rsa_head_outputs/`, `paper_figures/`, `tests/`. Its README lists the script behind each paper figure |
| `slurm_scripts/` | the SLURM launchers we ran, with the exact arguments (n, seeds, time limits). See its README |

Every module has a docstring with a usage line. The shared condition vocabulary is in
`condition_registry.py`.

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


Included alongside the CMA cells:

| path | content |
|---|---|
| `results/aggregated/` | fingerprints, cross-condition correlations and tests, head sets, head-selection inventory, path-patching and head-domain analyses |
| `results/ablation/` | cumulative and group ablation CSVs, random baselines |
| `results/circuit/` | Stage-1 configs and every Stage-2 steering run (`per_prompt.csv`, `summary.csv`, `attention.csv`) |
| `results/rsa_head_outputs/` | stimulus grids, RDM/Mantel tables, DLA, state geometry, per-head steering |
| `results/lookback_analysis/` | attention-mass tables per condition |
| `results/summary/` | the statistics behind the paper's tables and the heatmap manifest |
| `results/tom_performance/` | behavioral accuracy runs |
