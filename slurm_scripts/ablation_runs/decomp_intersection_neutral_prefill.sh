#!/bin/bash -l
#SBATCH --job-name=abl_decomp_inter_neutral
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --partition=preempt_lsong,gpuq,a100_preemptable,h200_preemptable,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_decomp_inter_neutral_%j.out
#SBATCH --error=logs/abl_decomp_inter_neutral_%j.err

# Fork of slurm_scripts/ablation_decomposition.sh, restricted to the
# intersection_belief_rank condition and swapped to the NEUTRAL system prompt
# (matches baseline_neutral_prefill.sh). Keeps per_variant prefill so this is
# comparable to the belief_only_neutral_prefill baseline.
#
# Drops the array: only intersection_belief_rank is needed here (K=19 heads
# ranked by mean_score, evaluated on false_photo + other variants).
#
# Prereq: results/lookback_analysis/_summary/decomposition_heads_end.csv must
# be present on HPC (build locally with
# `python analysis_scripts/build_decomposition_head_csvs.py` and scp it over -- the
# build script's inputs are excluded from sync_to_hpc.sh).

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

HEAD_CSV=results/lookback_analysis/_summary/decomposition_heads_end.csv
if [ ! -f "$HEAD_CSV" ]; then
  echo "ERROR: $HEAD_CSV not present on HPC. Build it locally and scp it over." >&2
  exit 1
fi

python cma_scripts/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --templates food_truck hair_styling library_book \
  --seed 1 \
  --n_devices 1 \
  --device_map cuda \
  --head_source per_condition \
  --head_csv $HEAD_CSV \
  --condition intersection_belief_rank \
  --rank_by mean_score \
  --eval_variants false_belief true_belief false_photo true_photo basic_ab basic_ba \
  --system_prompt "You are a helpful assistant. Answer concisely." \
  --assistant_prefill per_variant \
  --run_label decomp_intersection_belief_rank_neutral_prefill \
  --output_dir results/ablation/decomp_intersection_belief_rank_neutral_prefill
