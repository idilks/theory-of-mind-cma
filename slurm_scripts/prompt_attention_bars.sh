#!/bin/bash
#SBATCH --job-name=prompt_attn_bars
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/prompt_attn_bars_%j.out
#SBATCH --error=logs/prompt_attn_bars_%j.err

# Token-level attention bars over a real prompt: C1 (divergence) vs C5 (retrieval),
# significant end-position CMA heads only, final-token ("lookback") query.
# CPU-only -- reads the existing attention.pt captures from lookback_capture_full.sh,
# no model load, no GPU. Tokenizer is read from the shared HF cache (offline).
#
# One figure per template x prompt type:
#   false-belief prompts: C1 base_rule ABA (ABA->ABB cell) vs C5 ABA->ABA cell
#   true-belief prompts:  C1 base_rule ABB (ABB->ABA cell) alone -- C5 has no
#                         true-belief prompts (both of its cells are false-belief)
# Outputs (pulled by pull_results.sh block 2; no .pt written):
#   results/lookback_analysis/_prompt_bars/C1_vs_C5_{template}_FB.png/.pdf
#   results/lookback_analysis/_prompt_bars/C1_{template}_TB.png/.pdf
#   ..._tokens.csv          per-token values (weighted mean + every significant head)
#   ..._region_summary.csv  per-head mass on first token / correct loc / other loc /
#                           other anchors / self / rest, over ALL 150 prompts

export LD_LIBRARY_PATH=/dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis/lib:$LD_LIBRARY_PATH
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HUB_OFFLINE=1

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis
mkdir -p logs

ROOT=results/lookback_analysis
OUT=$ROOT/_prompt_bars
TEMPLATES=("food_truck" "hair_styling" "library_book")

for TEMPLATE in "${TEMPLATES[@]}"; do
  C1_FB=$ROOT/abstract_belief/template_${TEMPLATE}/base_rule_ABA_exp_rule_ABB/patch_before_movement
  C1_TB=$ROOT/abstract_belief/template_${TEMPLATE}/base_rule_ABB_exp_rule_ABA/patch_before_movement
  C5_FB=$ROOT/answer_changes_location_only/template_${TEMPLATE}/base_rule_ABA_exp_rule_ABA/patch_before_movement

  if [ -f "$C1_FB/attention.pt" ] && [ -f "$C5_FB/attention.pt" ]; then
    echo "=== $TEMPLATE false-belief: C1 vs C5 ==="
    python scripts/lookback_analysis/prompt_attention_bars.py \
      --capture_dir "$C1_FB" --label "C1 divergence heads, $TEMPLATE, false-belief prompt" \
      --capture_dir "$C5_FB" --label "C5 retrieval heads, $TEMPLATE, false-belief prompt" \
      --out_stem "$OUT/C1_vs_C5_${TEMPLATE}_FB"
  else
    echo "MISSING attention.pt for $TEMPLATE false-belief (C1: $C1_FB, C5: $C5_FB)"
  fi

  if [ -f "$C1_TB/attention.pt" ]; then
    echo "=== $TEMPLATE true-belief: C1 ==="
    python scripts/lookback_analysis/prompt_attention_bars.py \
      --capture_dir "$C1_TB" --label "C1 divergence heads, $TEMPLATE, true-belief prompt" \
      --out_stem "$OUT/C1_${TEMPLATE}_TB"
  else
    echo "MISSING attention.pt: $C1_TB"
  fi
done

ls -la $OUT
