#!/bin/bash
#SBATCH --job-name=abl_c1c5_full_on_code
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
#SBATCH --partition=h200_preemptable,preempt_wager
#SBATCH --array=0-15%8
#SBATCH --mem=50GB
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_c1c5_full_on_code_%A_%a.out
#SBATCH --error=logs/abl_c1c5_full_on_code_%A_%a.err

# Yang et al. 2025 §3.4/B.5-style cumulative
# ablation of ALL 1920 heads, ranked by the ToM CMA fingerprints, evaluated
# on code AND ToM variants in one sweep — H1 dissociation ported to code,
# plus the within-domain Yang curve for free from the same ablation order.
#   tom_c1 = all heads ordered by the C1 (abstract_belief) fingerprint
#   tom_c5 = all heads ordered by the C5 (answer_changes_location_only) one
# Control: rank-aligned same-layer lowest-fingerprint-score heads
# (--control_source ranking). k grid: every k to 128, then stride 16 to
# 1920 (241 sizes), n=75/variant.
#
# TIMING (measured from the failed 9023133 array, July 17): ~4.85 min per
# k step (375 prompt evals), and the 241-step grid runs TWICE (target +
# control curves) -> ~39 h per config, vs the 11:59 wall limit. So each
# config is split 4 ways: 2 curves x 2 k-grid chunks ([0:121) and
# [121:241)), ~10 h each -> 16 array jobs. ablation_experiment.py now
# checkpoints per k step and resumes, so a timeout or preemption
# (--requeue) costs at most one step — just let the job requeue or
# resubmit the same array index.
#
# AFTER ALL 4 CHUNKS OF A CONFIG FINISH, merge them:
#   python analysis_scripts/merge_cumulative_ablation_chunks.py \
#     --chunks results/ablation/$LABEL/chunk_* \
#     --output_dir results/ablation/$LABEL --run_label $LABEL
#
# Prereq on HPC: results/aggregated/code_head_sets/tom_c1_c5_all_heads_ranked.csv
# (build locally with `python analysis_scripts/build_tom_c1_c5_full_ranking_csv.py`;
#  results/aggregated syncs via sync_to_hpc.sh.)

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

HEAD_CSV=results/aggregated/code_head_sets/tom_c1_c5_all_heads_ranked.csv
if [ ! -f "$HEAD_CSV" ]; then
  echo "ERROR: $HEAD_CSV not present on HPC. Run analysis_scripts/build_tom_c1_c5_full_ranking_csv.py locally; sync_to_hpc.sh carries it over." >&2
  exit 1
fi

NEUTRAL_PROMPT="You are a helpful assistant. Answer concisely."
MATCHED_PROMPT="You are a helpful assistant reasoning about agents' beliefs."

# IDX decomposition: 16 jobs = config (4) x curve (2) x chunk (2)
IDX=$SLURM_ARRAY_TASK_ID
CFG=$(( IDX / 4 ))
CURVE_I=$(( (IDX / 2) % 2 ))
CHUNK_I=$(( IDX % 2 ))

HEADSETS=(  "tom_c1"          "tom_c1"          "tom_c5"          "tom_c5" )
PREFILLS=(  "none"            "per_variant"     "none"            "per_variant" )
SYSPROMPTS=( "$NEUTRAL_PROMPT" "$MATCHED_PROMPT" "$NEUTRAL_PROMPT" "$MATCHED_PROMPT" )

CURVES=(  "target" "control" )
K_STARTS=( 0   121 )
K_ENDS=(   121 241 )

HEADSET=${HEADSETS[$CFG]}
PREFILL=${PREFILLS[$CFG]}
SYSPROMPT=${SYSPROMPTS[$CFG]}
CURVE=${CURVES[$CURVE_I]}
K_START=${K_STARTS[$CHUNK_I]}
K_END=${K_ENDS[$CHUNK_I]}

LABEL="${HEADSET}_full1920_on_code_tom_prefill_${PREFILL}"
CHUNK_DIR="results/ablation/$LABEL/chunk_${CURVE}_${K_START}_${K_END}"

echo "Job $IDX: headset=$HEADSET prefill=$PREFILL curve=$CURVE k-chunk=[$K_START:$K_END)"

python cma_scripts/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 75 \
  --max_new_tokens 8 \
  --seed 1 \
  --n_devices 1 \
  --device_map cuda \
  --head_source per_condition \
  --head_csv $HEAD_CSV \
  --condition $HEADSET \
  --rank_by mean_score \
  --control_source ranking \
  --k_dense_until 128 \
  --k_stride 16 \
  --curves $CURVE \
  --k_index_start $K_START \
  --k_index_end $K_END \
  --eval_variants code_copy code_alias false_belief true_belief basic_ab \
  --assistant_prefill $PREFILL \
  --system_prompt "$SYSPROMPT" \
  --run_label $LABEL \
  --output_dir $CHUNK_DIR
