#!/bin/bash -l
#SBATCH --job-name=steer_sel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager,preempt_lsong
#SBATCH --mem=120GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/steer_sel_%j.out
#SBATCH --error=logs/steer_sel_%j.err

# Head-output steering (paper Fig. 11) under a head-selection rule, all three
# target domains (belief, code, photo) in one run.
#
#   sbatch slurm_scripts/steer_state_axes_selection.sh sig4of6   # primary
#   sbatch slurm_scripts/steer_state_axes_selection.sh sig3of6   # sensitivity
#   sbatch slurm_scripts/steer_state_axes_selection.sh top32     # harness check
#
# The top32 run must reproduce steering_qwen/steer_summary.csv (tom, code) to
# three decimals and §SG5's belief numbers (u_belief +20.901, u_cm +15.638);
# the photo-domain rand_matched draws differ from the lost steer_photo_axes.py
# harness, so photo values are expected to agree only approximately.
#
# Inputs on the cluster (synced by push.sh): results/aggregated/head_selection/
# (built locally by analysis_scripts/build_significance_head_sets.py), the all-heads
# captures activations_full/, activations_full_photo/,
# activations_code_position_full/, and stimuli/, stimuli_photo/.
#
# Saves only at the end (240 items x 16 steered passes), so a preempted or
# timed-out job must be resubmitted. Walltime is an allocation limit, not a
# runtime estimate.
#
# Output: results/rsa_head_outputs/steering_qwen_${SELECTION}_tom-code-photo/
# Success: the log says "all exact: True" and prints the summary table.

set -eo pipefail
SELECTION="${1:-${SELECTION:?usage: sbatch $0 {sig4of6|sig3of6|top32}}}"
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
PROJECT_ROOT="${PROJECT_ROOT:-/dartfs/rc/lab/F/FranklandS/tom}"
conda activate "${TOM_ENV:-$PROJECT_ROOT/envs/tom_analysis}"
set -u
cd "$PROJECT_ROOT"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1

for f in results/aggregated/head_selection/comparison.csv \
         results/rsa_head_outputs/activations_full/food_truck.pt \
         results/rsa_head_outputs/activations_full_photo/food_truck.pt \
         results/rsa_head_outputs/activations_code_position_full/Qwen2.5-14B-Instruct/code_position.pt \
         results/rsa_head_outputs/stimuli/food_truck.csv \
         results/rsa_head_outputs/stimuli_photo/food_truck.csv; do
  [[ -s "$f" ]] || { echo "Missing input: $f" >&2; exit 1; }
done

echo "selection=$SELECTION job=$SLURM_JOB_ID node=$(hostname)"
python -u analysis_scripts/rsa_head_outputs/steer_state_axes.py \
  --device cuda --selection "$SELECTION" --domains tom code photo --check_inputs
python -u analysis_scripts/rsa_head_outputs/steer_state_axes.py \
  --device cuda --selection "$SELECTION" --domains tom code photo
