# analysis scripts

post-hoc analysis tools for causal mediation experiments.

## pooled statistical analysis

`analyze_pooled_conditions.py` performs rigorous statistical testing across multiple experimental conditions and templates.

### workflow

**step 1: run all individual experiments**
```bash
cd codebase/tasks/causal_analysis

# cross-belief condition (abstract context)
python cma.py \
  --use_template_system \
  --context_type abstract \
  --base_rule ABA \
  --prompt_num 50 \
  --model_type Qwen/Qwen2.5-14B-Instruct \
  --template_names food_truck hair_styling object_relocation

# within-belief condition (token context)
python cma.py \
  --use_template_system \
  --context_type token \
  --base_rule ABA \
  --prompt_num 50 \
  --model_type Qwen/Qwen2.5-14B-Instruct \
  --template_names food_truck hair_styling object_relocation

# control condition
python cma.py \
  --use_template_system \
  --context_type control \
  --base_rule ABA \
  --prompt_num 50 \
  --model_type Qwen/Qwen2.5-14B-Instruct \
  --template_names food_truck hair_styling object_relocation
```

each run produces `causal_scores_per_sample.pt` in its result folder.

**step 2: run pooled analysis**
```bash
python scripts/analyze_pooled_conditions.py \
  --results_dir results/causal_analysis/cma \
  --output_dir results/pooled_analysis \
  --n_permutations 1000 \
  --alpha 0.05
```

### what it does

1. **discovers** all `causal_scores_per_sample.pt` files in results directory
2. **groups** by condition (abstract → cross_belief, token → within_belief, etc)
3. **pools** across templates (combines food_truck + hair_styling + object_relocation)
4. **tests** each pooled condition with permutation testing + FDR correction
5. **compares** conditions with differential testing (e.g., cross_belief vs control)

### output structure

```
results/pooled_analysis/
├── pooled_by_condition/
│   ├── cross_belief_pooled_stats.csv        # all heads, pooled across templates
│   ├── cross_belief_significant_heads.csv   # only significant heads
│   ├── within_belief_pooled_stats.csv
│   ├── within_belief_significant_heads.csv
│   ├── control_pooled_stats.csv
│   └── control_significant_heads.csv
├── differential_tests/
│   ├── cross_belief_vs_control_differential.csv
│   ├── cross_belief_vs_control_significant_differential.csv
│   ├── cross_belief_vs_within_belief_differential.csv
│   └── cross_belief_vs_within_belief_significant_differential.csv
└── pooled_analysis_summary.txt              # human-readable summary
```

### interpreting results

**pooled condition tests** answer: "which heads are significant for this condition?"
- `cross_belief_significant_heads.csv`: heads that causally matter for belief-type distinction
- column `observed_score`: mean causal mediation effect
- column `q_value`: FDR-corrected p-value

**differential tests** answer: "which heads are SPECIFIC to this condition?"
- `cross_belief_vs_control_significant_differential.csv`: heads more important for cross-belief than control
- column `differential_effect`: difference in causal scores between conditions
- column `q_value`: FDR-corrected p-value for the difference

### for your paper

claim: "layers 29-35, heads 30-34 support belief tracking"

**statistical support**:
1. open `cross_belief_vs_control_significant_differential.csv`
2. filter for layers 29-35, heads 30-34
3. verify `q_value < 0.05` and `differential_effect > 0`
4. report: "N heads in layers 29-35 showed significantly greater activation during cross-belief patching compared to control (permutation test, FDR-corrected p < 0.05, differential effect = X.XX)"

### advanced usage

**custom differential tests**:
```bash
python scripts/analyze_pooled_conditions.py \
  --results_dir results/ \
  --differential_tests cross_belief:control cross_belief:within_belief control:within_belief
```

**more permutations for higher precision**:
```bash
python scripts/analyze_pooled_conditions.py \
  --results_dir results/ \
  --n_permutations 10000 \
  --alpha 0.01
```

**analyze specific conditions only**:
```bash
python scripts/analyze_pooled_conditions.py \
  --results_dir results/ \
  --conditions cross_belief control
```
