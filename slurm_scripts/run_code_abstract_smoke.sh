#!/bin/bash
#SBATCH --job-name=cma_code_abstract_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_lsong,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/code_abstract_smoke_%j.out
#SBATCH --error=logs/code_abstract_smoke_%j.err

export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

# Smoke test for code_abstract (C9): same-answer copy-vs-alias condition
# (indexed_list redesign). 5 prompts, end-patch, no permutation tests.
# Goal: verify pipeline runs end-to-end and produces non-zero causal scores.

python codebase/tasks/causal_analysis/cma.py \
  --condition code_abstract \
  --base_rule ABA \
  --template_names indexed_list \
  --seed 0 \
  --prompt_num 5 \
  --min_valid_sample_num 1 \
  --max_new_tokens 30 \
  --activation_name z \
  --model_type Qwen2.5-14B-Instruct \
  --no_wandb
