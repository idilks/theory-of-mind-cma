#!/bin/bash
#SBATCH --job-name=rsa_midpad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=07:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --array=0-1
#SBATCH --mem=110GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/rsa_midpad_%A_%a.out
#SBATCH --error=logs/rsa_midpad_%A_%a.err

# Init-vs-mutation pad grid: the direct test of CONDITION-SPECIFIC targeting.
#
# In the Exp 3 pad-twins grid the prefix pad moves BOTH candidate lookback
# targets (the line-1 `a = [...]` init literal and the mutation literal) by
# the same amount, and the suffix pad moves neither. So that grid cannot
# distinguish "points at the init literal" from "points at the mutation
# literal" — it only shows that something positional moves.
#
# This grid puts the second pad BETWEEN the definitions and the mutation
# line, so it moves the mutation literal and leaves the init literal fixed
# (verified: within a twin group, slot_char_pos is constant across mid-pad
# levels while mut_char_pos shifts by ~88 chars per 16 pad tokens).
#
# The prediction that discriminates a pointer from a tag or a single shared
# carrier is an INTERACTION:
#   copy + untouched queries read the init literal
#     -> sensitive to the prefix pad, FLAT in the mid pad
#   alias + mutated queries read the mutation literal
#     -> sensitive to BOTH
# A veracity tag predicts no pad sensitivity at all; one shared position
# carrier predicts equal sensitivity everywhere; only a target-specific
# pointer predicts the interaction.
#
# Same stimulus count as Exp 3 (48 cells x 2 content x 9 pad combos = 864),
# same analysis code — analyze_code_padtwins_rdms.py and
# padtwins_projection_test.py read the second pad factor from
# `suffix_pad_len`, which the builder fills with the mid-pad length here
# (mid_pad_len and pad_axis carry the truth; see the builder docstring).
#
# Analysis after pulling:
#   python scripts/rsa_head_outputs/analyze_code_padtwins_rdms.py \
#       --model qwen --skip_mantel      # pad-twin test on the mid grid
# and compare rho_prefix|suffix (= prefix response) against
# rho_suffix|prefix (= mid response) split by queried_object/slot_type.

export LD_LIBRARY_PATH=/dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis/lib:$LD_LIBRARY_PATH
export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd
mkdir -p .conda/pkgs/cache .conda/envs

cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

STIM=results/rsa_head_outputs/stimuli_code_padtwins
OUT=results/rsa_head_outputs/activations_code_midpad_full

# builder runs the same hard confound + pad-orthogonality checks as Exp 3;
# a nonzero exit means the grid is bad and nothing should be captured
if [ ! -f $STIM/code_midpad.csv ]; then
  python scripts/rsa_head_outputs/build_code_padtwins_stimuli.py \
    --pad_axis mid || exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  echo "Task 0: code_midpad capture, Qwen2.5-14B-Instruct"
  python scripts/rsa_head_outputs/capture_head_outputs_full.py \
    --template code_midpad \
    --stim_dir  $STIM \
    --out_dir   $OUT/Qwen2.5-14B-Instruct \
    --model_type Qwen2.5-14B-Instruct \
    --system_prompt code \
    --decode_answers \
    --capture_attn_summary
else
  echo "Task 1: code_midpad capture, gemma-2-27b-it"
  python scripts/rsa_head_outputs/capture_head_outputs_full.py \
    --template code_midpad \
    --stim_dir  $STIM \
    --out_dir   $OUT/gemma-2-27b-it \
    --model_type gemma-2-27b-it \
    --ungroup_grouped_query_attention \
    --system_prompt code \
    --decode_answers \
    --capture_attn_summary
fi
