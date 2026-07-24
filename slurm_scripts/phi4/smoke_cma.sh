#!/bin/bash -l
#SBATCH --job-name=phi4_smoke     # end-to-end compatibility + behavioral gate check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager,preempt_lsong
#SBATCH --mem=60GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/phi4_smoke_%j.out
#SBATCH --error=logs/phi4_smoke_%j.err

export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="***REMOVED_HF_TOKEN***"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd
mkdir -p .conda/pkgs/cache .conda/envs

cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

# phi-4 SMOKE TEST (July 2026) — run BEFORE submitting c1_cma.sh.
# GENERATION-ONLY: --min_valid_sample_num 13 > --prompt_num 12 is an
# unreachable threshold, so cma.py's STEP 2 early-returns after the
# behavioral evaluation and NEVER enters the patching loop (~20 min total,
# mostly model load). Verifies in one job:
#   1. TL loading of microsoft/phi-4 + GQA ungrouping (40H / 10 KV heads)
#   2. ChatML template with system role, end-to-end through the generator
#   3. behavioral gate: per-prompt base_passes/exp_passes at gen_acc >= 0.9;
#      if phi-4 can't do the ToM task (cf. Llama-3-8B at 80% BigToM),
#      it shows here, not 30 GPU-h in
# READ THE RESULT in logs/phi4_smoke_<jobid>.out:
#   grep "base_passes\|Valid prompts for CMA" — want >= ~10/12 valid.
#   Expected final lines: "Not enough valid samples: N < 13" — that is the
#   intended early exit, NOT a failure. N is the number you care about.
# (Patch-path note: end-position patching itself is first exercised by the
#  C1 array — to de-risk, release index 0 alone first:
#  sbatch --array=0 slurm_scripts/phi4/c1_cma.sh, check causal_scores.pt
#  appears, then submit the rest with sbatch --array=1-5%4.)
# n=12 result lands under sample_num_12_* — max-sample_num pickers ignore it.

python codebase/tasks/causal_analysis/cma.py \
  --use_template_system \
  --condition abstract_belief \
  --base_rule ABA \
  --template_names food_truck \
  --prompt_num 12 \
  --min_valid_sample_num 13 \
  --max_new_tokens 5 \
  --activation_name z \
  --model_type phi-4 \
  --ungroup_grouped_query_attention \
  --question_style instruction \
  --no_statistical_tests
