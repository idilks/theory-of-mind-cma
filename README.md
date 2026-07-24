# modularSymbTom

causal mediation analysis (CMA) of theory of mind in transformer language models. we patch
activations between matched prompt conditions to identify the attention heads responsible for
belief tracking, location retrieval, and state tracking. built on the LLMSymbMech framework
(ICML 2025, vendored under `codebase/`).

- **primary model:** Qwen2.5-14B-Instruct (48 layers × 40 heads = 1920 heads)
- **replication arm:** Gemma-2-27B-it (`ungroup_gqa` path)
- **templates:** food_truck, hair_styling, library_book (+ a code-domain `indexed_list` arm)

## setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN="your_hf_token"   # https://huggingface.co/settings/tokens
```

pinned versions matter: `transformers==4.46.3`, `transformer-lens==2.13.0` (see `requirements.txt`).

## quick start — run a CMA

```bash
cd codebase/tasks/causal_analysis

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

- `--patch_after_movement` selects the belief-formation patch position; omitting it patches the
  end (final) token. **all headline results use the end position.**
- statistical testing is a max-statistic permutation test with FWER correction (5000 perms);
  the shared implementation lives in `codebase/tasks/causal_analysis/statistical_testing.py` —
  import it, don't reimplement.

## conditions (canonical C-numbering)

the canonical numbering follows `scripts/aggregate_conditions.py`. older scripts use other
numberings — when in doubt, refer to a condition by its folder name.

| #  | folder name                         | what varies (base↔exp)          | answers      | what it finds                          |
|----|-------------------------------------|---------------------------------|--------------|----------------------------------------|
| C1 | `abstract_belief`                   | belief type (false↔true)        | same (a,a)   | belief-divergence heads                |
| C2 | `abstract_photo`                    | photo timing (false↔true)       | same (a,a)   | state-divergence heads (no agent)      |
| C3 | `answer_changes_belief`             | belief type (false↔true)        | different    | belief + answer heads (= C1 + C5)      |
| C4 | `answer_changes_photo`              | photo timing (false↔true)       | different    | state + answer heads                   |
| C5 | `answer_changes_location_only`      | location tokens only, ToM frame | different    | answer / location-retrieval heads      |
| C6 | `answer_changes_location_only_photo`| location tokens only, photo     | different    | answer / location-retrieval heads      |
| C7 | `state_only_no_belief`              | state change, no belief         | different    | state-update heads (control)           |

a parallel **code-domain arm** (`code_abstract`, `code_answer_changes`, `code_control`) uses
copy-vs-alias list mutation as a non-ToM divergence analog. the `CMACondition` enum in
`cma_config.py` is the source of truth for all condition members — import it rather than
hand-rolling condition tables.

## repo layout

| path                              | what's here                                                            |
|-----------------------------------|-----------------------------------------------------------------------|
| `codebase/`                       | vendored LLMSymbMech framework; `tasks/causal_analysis/` is the CMA pipeline (`cma.py`, `ablation_experiment.py`, `cma_config.py`, prompt generators) |
| `scripts/`                        | post-hoc analysis: cross-condition stats, heatmaps, ablation, RSA/DLA, lookback. see `scripts/README.md` and `scripts/condition_registry.py` (canonical C1–C7 registry) |
| `slurm_scripts/`                  | HPC batch jobs (CMA, ablation, RSA); per-model subdirs (`gemma/`, …)   |
| `tom_datasets/`                   | prompt templates and stimulus data                                    |
| `sync_to_hpc.sh` / `pull_results.sh` | move code up to the cluster / pull results back                    |

## result path structure

```
results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct/
  {condition}/template_{name}/base_rule_{base}_exp_rule_{exp}/
    {patch_position}/                          # patch_after_movement (bf) or patch_before_movement (end)
      z_seed_{S}_shuffle_False/logit/sample_num_{N}_gen_acc_0.9/
        causal_scores.pt, heatmap.png, statistical_test_results.csv
```

seed priority when resolving a source: `combined` > `1` > `0`. use **seed=1, end position** for
the full 7-condition analysis (only seed with complete coverage). large result trees
(`results/`, `other_results/`) are gitignored — they're pulled from the cluster, not committed.

## notes

- `codebase/tasks/identity_rules/` is the upstream in-context-learning task; the ToM prompt
  generators still import from it, so it's a dependency, not dead code.
- research writeups, design docs, and PI-facing reports are kept private (gitignored) and are not
  part of this shared repo.
