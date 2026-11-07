#!/bin/bash
#SBATCH --job-name=reanalyze_c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --partition=standard
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/reanalyze_c4_%j.out
#SBATCH --error=logs/reanalyze_c4_%j.err

# Re-run analyze_lookback_attention.py + per_token_profile.py on C4's existing
# attention.pt blobs. C4's lookback_scores.csv files are from an old version of
# the analyze script (legacy `state` column only, no state_first/state_second/
# state_correct/info_loss/char_q/belief_form, no agent_first/agent_last queries).
# Re-running analyze on the existing tensors regenerates the CSVs with the
# current full schema. CPU-only -- no model load, no GPU.

export LD_LIBRARY_PATH=/dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis/lib:$LD_LIBRARY_PATH

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

CONDITION="answer_changes_photo"
TEMPLATES=("food_truck" "hair_styling" "library_book")
DIRECTIONS=("ABA_exp_rule_ABA" "ABB_exp_rule_ABB")

for TEMPLATE in "${TEMPLATES[@]}"; do
  for DIR in "${DIRECTIONS[@]}"; do
    CAP_DIR="results/lookback_analysis/${CONDITION}/template_${TEMPLATE}/base_rule_${DIR}/patch_before_movement"
    if [ ! -f "$CAP_DIR/attention.pt" ]; then
      echo "MISSING: $CAP_DIR/attention.pt -- skipping"
      continue
    fi
    echo "=== $CAP_DIR ==="
    python analysis_scripts/lookback_analysis/analyze_lookback_attention.py --capture_dir "$CAP_DIR"
    python analysis_scripts/lookback_analysis/per_token_profile.py --capture_dir "$CAP_DIR" --top_k 5 --no_tokenizer
  done
done

echo "done."
