#!/bin/bash
#SBATCH --job-name=abl_decomp_c1c5_code
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --array=0-5%6
#SBATCH --mem=50GB
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_decomp_c1c5_code_%A_%a.out
#SBATCH --error=logs/abl_decomp_c1c5_code_%A_%a.err

# C1/C5 set-subtraction ablation ON CODE (July 2026) — the thesis H1
# deconfounding (run5/6/7, May 2026, ToM evals only) ported to the code
# domain. Motivation: in the full-1920 cumulative sweep the C5-ordering
# flip on code_copy turns on exactly where the C1 core enters the C5
# ranking (C5 positions 11/13/17 = C1 ranks 8/5/2; flip 0.006 at k=8 ->
# 0.169 at k=16), so the cumulative curves cannot separate "C1 = divergence
# encoding" from "C5 = retrieval". Fixed-partition sets can:
#   a_only        = T_C1 \ T_C5  (21 belief-unique heads)
#   b_only        = T_C5 \ T_C1  (11 loc-unique heads)
#   intersection  = T_C1 ∩ T_C5  (11 shared heads)
# Prediction if the deconfounding transfers: a_only + intersection flip
# code_copy (alias spared); b_only does NOT flip code_copy but degrades
# answer production generally (both code variants + basic_ab).
#
# Pair/CSV/sets pinned to exactly what run5/6/7 used
# (condition_overlap_heads_end.csv, pair "abstract_belief,
# answer_changes_location_only"); control = same-layer lowest-|score|
# heads from the wandb permutation CSVs (default --control_source), also
# identical to the May runs. Only the eval side changes: code_copy /
# code_alias join the ToM variants, prompt_num=75 to match the full-1920
# sweep, both eval protocols. k sweeps 1..|set| (dense), so each run also
# yields the within-set mini cumulative curve; the headline number is the
# final k. No chunking — nothing to merge; per-k checkpoint + --requeue
# make preemptions cheap.
#
# Timing: worst set (a_only, 21 heads) = 22 k-steps x 2 curves x 375 evals
# at ~4.85 min/step ≈ 3.6 h; 11-head sets ≈ 2 h. Fits 5:59.
#
# Prereq on HPC: results/lookback_analysis/_summary/condition_overlap_heads_end.csv
# (same file run5/6/7_matched_prefill expect; local fuller copy lives at
# results/lookback_analysis/_summary/).

export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd
mkdir -p .conda/pkgs/cache .conda/envs

cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis
export LD_LIBRARY_PATH=/dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis/lib:$LD_LIBRARY_PATH
mkdir -p logs

HEAD_CSV=results/lookback_analysis/_summary/condition_overlap_heads_end.csv
if [ ! -f "$HEAD_CSV" ]; then
  echo "ERROR: $HEAD_CSV not present on HPC (run5/6/7_matched_prefill expect the same file)." >&2
  exit 1
fi

NEUTRAL_PROMPT="You are a helpful assistant. Answer concisely."
MATCHED_PROMPT="You are a helpful assistant reasoning about agents' beliefs."

# IDX decomposition: 6 jobs = set (3) x prefill (2)
IDX=$SLURM_ARRAY_TASK_ID
SET_I=$(( IDX / 2 ))
PRE_I=$(( IDX % 2 ))

SETS=(   "a_only"           "b_only"          "intersection" )
LABELS=( "belief_minus_loc" "loc_minus_belief" "shared_belief_loc" )
PREFILLS=( "none" "per_variant" )

SET=${SETS[$SET_I]}
SETLABEL=${LABELS[$SET_I]}
PREFILL=${PREFILLS[$PRE_I]}
if [ "$PREFILL" = "none" ]; then SYSPROMPT="$NEUTRAL_PROMPT"; else SYSPROMPT="$MATCHED_PROMPT"; fi

LABEL="decomp_${SETLABEL}_on_code_prefill_${PREFILL}"

echo "Job $IDX: set=$SET ($SETLABEL) prefill=$PREFILL"

python cma_scripts/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 75 \
  --max_new_tokens 8 \
  --templates food_truck hair_styling library_book \
  --seed 1 \
  --n_devices 1 \
  --device_map cuda \
  --head_source overlap_set \
  --head_csv $HEAD_CSV \
  --pair "abstract_belief,answer_changes_location_only" \
  --set $SET \
  --eval_variants code_copy code_alias false_belief true_belief basic_ab \
  --system_prompt "$SYSPROMPT" \
  --assistant_prefill $PREFILL \
  --run_label $LABEL \
  --output_dir results/ablation/$LABEL
