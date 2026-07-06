#!/bin/bash
#SBATCH --job-name=abl_xframe_photo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --partition=preempt_lsong,gpuq,a100_preemptable,h200_preemptable,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_xframe_photo_%j.out
#SBATCH --error=logs/abl_xframe_photo_%j.err

# Corrected cross-frame TOP ROW for ablation_cross_frame_3col figure.
# Mirrors Run B's protocol (matched sysprompt + per_variant prefill) but
# uses C2-derived (abstract_photo) heads instead of C1-derived heads.
# Paired with belief_only_sysprompt_prefill/ (Run B), which already
# provides the bottom row (belief heads ablation -> false_photo eval).
#
# Same seed/prompt_num as Run B so the eval set is identical across the
# two rows of the figure.

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

python codebase/tasks/causal_analysis/ablation_experiment.py \
  --model_type Qwen2.5-14B-Instruct \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --templates food_truck hair_styling library_book \
  --seed 1 \
  --n_devices 1 \
  --device_map cuda \
  --head_source per_condition \
  --head_csv results/lookback_analysis/_summary/top20_by_condition_end.csv \
  --condition abstract_photo \
  --rank_by max_observed_score \
  --eval_variants false_belief true_belief false_photo true_photo basic_ab basic_ba \
  --system_prompt "You are a helpful assistant reasoning about agents' beliefs." \
  --assistant_prefill per_variant \
  --run_label photo_heads_sysprompt_prefill \
  --output_dir results/ablation/photo_heads_sysprompt_prefill
