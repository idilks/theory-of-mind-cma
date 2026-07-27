#!/bin/bash
#SBATCH --job-name=circuit_s2_ctl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=07:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --array=0-1
#SBATCH --mem=90GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/circuit_s2_ctl_%A_%a.out
#SBATCH --error=logs/circuit_s2_ctl_%A_%a.err

# Two follow-up steering jobs asked for in writeups/results_for_pi/steering-only.qmd.
# Both reuse the Stage-1 directions and the SAME --control_seed 99, so the
# control directions are byte-identical to the ones already in L41_main.
#
#   task 0  ATTENTION NULL ("lets do the null").
#           Reruns the random / heads_ctl / shuffled legs on held-out
#           false_belief + true_belief prompts with the reader-head attention
#           readout ON. Goes to a FRESH output dir (L41_attn_null) because the
#           resume logic skips configs already present in L41_main/per_prompt.csv
#           — writing into L41_main would silently skip every config we want.
#           Consistency check after the run: the flip scores in
#           L41_attn_null/summary.csv must reproduce L41_main's control column;
#           if they don't, the control directions were not reproduced.
#           2 prompt sets x (1 clean + 3 directions x 4 alphas) = 26 passes
#           x 225 prompts = 5850 forwards, all with pattern caching. ~2-3 h.
#
#   task 1  NEGATIVE-CONTROL LEGS (TB-delete / FB-inject).
#           Resumes L41_main and adds only the configs that are missing:
#           false_belief x inject x {target_topk, target_rank1} x 4 alphas.
#           (true_belief x delete already ran in the original submission —
#           it is in L41_main/summary.csv, just unreported.) Attention is
#           captured automatically for these (target direction on a
#           belief prompt set). 8 configs x 225 prompts = 1800 forwards, <1 h.
#           Everything else in L41_main is skipped by the resume logic.
#
# After BOTH finish, pull and run on the laptop:
#   python scripts/circuit/analyze_steering_attention_null.py
# which merges L41_main (clean + target) with L41_attn_null (controls) and
# emits the target-vs-control attention comparison the qmd is missing.

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

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  echo "Task 0: attention null (control directions, attention capture ON)"
  python scripts/circuit/stage2_steering.py \
    --site L41 \
    --run_label attn_null \
    --only_controls \
    --attn_all_directions \
    --control_seed 99
else
  echo "Task 1: negative-control legs (FB-inject; TB-delete already present)"
  python scripts/circuit/stage2_steering.py \
    --site L41 \
    --run_label main \
    --skip_cross_domain \
    --control_seed 99
fi
