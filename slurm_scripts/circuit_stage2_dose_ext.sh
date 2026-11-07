#!/bin/bash
#SBATCH --job-name=circuit_s2_dose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --mem=90GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/circuit_s2_dose_%j.out
#SBATCH --error=logs/circuit_s2_dose_%j.err

# Steering-coefficient extension for the layer-41 residual edit (Nov 2026,
# camera-ready). Motivates the paper's alpha = 4 by showing (a) what the
# belief legs do above the original grid {0.5, 1, 2, 4} and (b) how much a
# belief-free prompt set is disturbed by the same edit at every alpha.
#
# Writes to a FRESH output dir (L41_dose_ext): the resume logic keys on
# (prompt_set, mode, direction, dose), so a resumed L41_main would be fine
# too, but keeping the frozen paper data untouched is worth the extra clean
# passes (which double as a reproduction check against L41_main's clean rows).
# Same --control_seed 99 as L41_main, so the random / heads_ctl / shuffled
# directions are byte-identical to the paper's controls.
#
#   Part A  belief legs (TB inject, FB delete, sign-reversal legs, basic_ab,
#           code_alias) at alpha in {6, 8, 12, 16}; rank-1 target + controls.
#           The 8-dim swap operator is skipped (off-manifold at alpha = 4).
#           ~12 legs x 4 doses x ~225 prompts ~ 10k forwards.
#   Part B  generic belief-free prompts (analysis_scripts/circuit/generic_prompts.py),
#           full grid {0.5 .. 16}, rank-1 inject + delete, three controls
#           inject; KL + top-1 agreement readouts. ~9k forwards.
#
# Both parts checkpoint per config and resume on requeue — just resubmit.
# Afterwards: pull_results.sh, then the dose-extension analysis on the laptop
# (merges L41_main doses <= 4 with L41_dose_ext doses >= 6 + generic).

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

if [ ! -f results/aggregated/circuit/stage1/directions_L41.npz ]; then
  echo "ERROR: stage-1 directions missing — run stage1_geometric_screen.py locally and sync." >&2
  exit 1
fi

set -e
echo "Part A: belief legs, alpha in {6, 8, 12, 16}, rank-1 + controls"
python analysis_scripts/circuit/stage2_steering.py \
  --site L41 \
  --run_label dose_ext \
  --control_seed 99 \
  --skip_topk \
  --doses 6 8 12 16

echo "Part B: generic collateral leg, full alpha grid"
python analysis_scripts/circuit/stage2_steering.py \
  --site L41 \
  --run_label dose_ext \
  --control_seed 99 \
  --only_generic \
  --doses 0.5 1 2 4 6 8 12 16
