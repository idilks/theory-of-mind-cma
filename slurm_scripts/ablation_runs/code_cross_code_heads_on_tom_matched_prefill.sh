#!/bin/bash
#SBATCH --job-name=abl_codeheads_tom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_codeheads_tom_%j.out
#SBATCH --error=logs/abl_codeheads_tom_%j.err

# Experiment 2b (cross-domain transfer, reverse direction): cumulative
# ablation of the code_abstract divergence heads (K=12), evaluated on ToM
# false_belief / true_belief prompts under the PRIMARY ToM protocol
# (belief-mentioning matched system prompt + per_variant prefill, baseline
# flip ≈ 0). basic_ab included as the
# retrieval control (H1-style specificity check). Uses only pre-existing
# pipeline paths — the new code here is just the head-list CSV.
#
# Prereq: results/aggregated/code_head_sets/code_heads_end.csv on HPC
# (build locally with `python analysis_scripts/build_code_head_csvs.py`, scp over).

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
  --condition code_abstract \
  --rank_by mean_score \
  --eval_variants false_belief true_belief basic_ab \
  --system_prompt "You are a helpful assistant reasoning about agents' beliefs." \
  --assistant_prefill per_variant \
  --run_label code_abstract_on_tom_matched_prefill \
  --output_dir results/ablation/code_cross_code_abstract_on_tom_matched_prefill
