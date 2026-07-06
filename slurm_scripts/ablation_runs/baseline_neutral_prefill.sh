#!/bin/bash
#SBATCH --job-name=abl_neutral_prefill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:59:00
#SBATCH --partition=preempt_lsong,gpuq,a100_preemptable,h200_preemptable,preempt_wager
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/abl_neutral_prefill_%j.out
#SBATCH --error=logs/abl_neutral_prefill_%j.err

# Diagnostic Run C: completes the 2x2 (sysprompt x prefill) decomposition.
#   thesis: neutral sysprompt + no prefill   -> phi(false_belief, k=0) ~ 0.37
#   Run A:  matched sysprompt + no prefill   -> 0.014
#   Run B:  matched sysprompt + prefill      -> 0.010
#   Run C:  neutral sysprompt + prefill      -> ?
#
# If Run C ~= 0.01: formatting (prefill) was the load-bearing fix; the
# system-prompt swap was not necessary. We can keep the neutral sysprompt
# in ablation eval, which removes the belief-priming asymmetry that would
# otherwise contaminate the H2 (C1 vs C2) test on photo evaluation.
#
# If Run C ~= 0.37: the system prompt is doing real work and we have to
# confront the priming confound directly (re-run CMA with a neutral-but-
# engaged prompt, or accept the asymmetry as a documented caveat).

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
  --condition abstract_belief \
  --rank_by max_observed_score \
  --eval_variants false_belief true_belief false_photo true_photo basic_ab basic_ba \
  --system_prompt "You are a helpful assistant. Answer concisely." \
  --assistant_prefill per_variant \
  --run_label belief_only_neutral_prefill \
  --output_dir results/ablation/belief_only_neutral_prefill
