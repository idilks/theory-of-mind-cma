#!/bin/bash
#SBATCH --job-name=abl_code_within
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --array=0-3%4
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_code_within_%A_%a.out
#SBATCH --error=logs/abl_code_within_%A_%a.err

# Experiments 1 + 4 of the code-domain ablation arm (July 2026 indexed_list):
#   1. Within-domain: cumulative ablation of the code_abstract divergence
#      heads (K=12, CMA-mean-score-ranked union over both directions),
#      evaluated on paired code_copy (false-belief analog) / code_alias
#      (true-belief analog) prompts.
#   4. Specificity contrast: same sweep for the code_control answer-writer
#      heads (K=14) — expected to hurt answer production generally rather
#      than divergence-specifically.
# Each condition runs under both eval protocols:
#   none        = plain generation (first token is already the merged 'X
#                 answer token, so p_correct/p_alt come for free)
#   per_variant = leading-quote prefill (single-token probe; next token is
#                 the bare answer char)
#
# Prereq: results/aggregated/code_head_sets/code_heads_end.csv on HPC
# (results/ is excluded from sync_to_hpc.sh — build locally with
# `python scripts/build_code_head_csvs.py` and scp the folder over).

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

HEAD_CSV=results/aggregated/code_head_sets/code_heads_end.csv
if [ ! -f "$HEAD_CSV" ]; then
  echo "ERROR: $HEAD_CSV not present on HPC. Build it locally and scp it over." >&2
  exit 1
fi

CONDITIONS=( "code_abstract" "code_abstract" "code_control" "code_control" )
PREFILLS=(   "none"          "per_variant"   "none"         "per_variant" )

IDX=$SLURM_ARRAY_TASK_ID
CONDITION=${CONDITIONS[$IDX]}
PREFILL=${PREFILLS[$IDX]}
LABEL="${CONDITION}_prefill_${PREFILL}"

echo "Job $IDX: condition=$CONDITION prefill=$PREFILL"

python codebase/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 150 \
  --max_new_tokens 8 \
  --seed 1 \
  --n_devices 1 \
  --device_map cuda \
  --head_source per_condition \
  --head_csv $HEAD_CSV \
  --condition $CONDITION \
  --rank_by mean_score \
  --eval_variants code_copy code_alias \
  --assistant_prefill $PREFILL \
  --run_label code_within_$LABEL \
  --output_dir results/ablation/code_within_$LABEL
