#!/bin/bash
#SBATCH --job-name=abl_tomheads_code
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --array=0-1%2
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_tomheads_code_%A_%a.out
#SBATCH --error=logs/abl_tomheads_code_%A_%a.err

# Experiment 2a (cross-domain transfer, the key one): cumulative ablation of
# the ToM C1∩C2 19-head intersection (from other_results/ablation/
# decomp_intersection_belief_rank, copied into the code_heads CSV as
# condition=tom_intersection), evaluated on CODE prompts (code_copy /
# code_alias). If the divergence machinery is domain-general, ablating the
# ToM intersection should degrade/flip code_copy the way it degrades
# false_belief. Both eval protocols (plain generation + quote prefill).
#
# Prereq: results/aggregated/code_head_sets/code_heads_end.csv on HPC
# (build locally with `python scripts/build_code_head_csvs.py`, scp over).

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

PREFILLS=( "none" "per_variant" )
PREFILL=${PREFILLS[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_ARRAY_TASK_ID: tom_intersection heads on code prompts, prefill=$PREFILL"

python codebase/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 150 \
  --max_new_tokens 8 \
  --seed 1 \
  --n_devices 1 \
  --device_map cuda \
  --head_source per_condition \
  --head_csv $HEAD_CSV \
  --condition tom_intersection \
  --rank_by mean_score \
  --eval_variants code_copy code_alias \
  --assistant_prefill $PREFILL \
  --run_label tom_intersection_on_code_prefill_$PREFILL \
  --output_dir results/ablation/code_cross_tom_intersection_prefill_$PREFILL
