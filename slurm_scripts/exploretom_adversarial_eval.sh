#!/bin/bash -l
#SBATCH --job-name=exploretom_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=h200_preemptable,preempt_wager
#SBATCH --mem=100GB
#SBATCH --time=05:59:00
#SBATCH --array=0-1
#SBATCH --output=logs/exploretom_%A_%a.out
#SBATCH --error=logs/exploretom_%A_%a.err

# Submit from the project root AFTER syncing scripts/ and slurm_scripts/:
#   mkdir -p logs
#   sbatch slurm_scripts/exploretom_adversarial_eval.sh
# 0 = Qwen2.5-14B-Instruct; 1 = Gemma-2-27B-it. Same frozen 100 prompts.
# H200 requested for both; time is a scheduling limit, not a runtime estimate.
# Re-submit to resume committed examples. No automatic model/dataset selection.
set -eo pipefail
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
PROJECT_ROOT="${PROJECT_ROOT:-/dartfs/rc/lab/F/FranklandS/tom}"
conda activate "${TOM_ENV:-$PROJECT_ROOT/envs/tom_analysis}"
set -u
cd "$PROJECT_ROOT"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# Same cache variables as the Gemma/Qwen CMA jobs, which already hold both
# models; do not depend on whatever the login profile happens to set.
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

case "${SLURM_ARRAY_TASK_ID:?Submit using sbatch}" in
  0) MODEL=qwen ;;
  1) MODEL=gemma
     # Gated repo: same token convention as slurm_scripts/gemma/*.sh (never printed).
     : "${HF_TOKEN:?HF_TOKEN not set - add it to ~/.bashrc on the HPC (see README setup)}"
     export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" ;;
  *) echo "Expected task 0 (Qwen) or 1 (Gemma)" >&2; exit 1 ;;
esac
DATA_DIR="${EXPLORETOM_DATA_DIR:-scripts/exploretom/data/adversarial_100_seed0}"
OUTPUT_ROOT="${EXPLORETOM_OUTPUT_ROOT:-results/exploretom/released_100_seed0}"
test -s "$DATA_DIR/prompts.jsonl"
test -s "$DATA_DIR/head_groups.json"
EXTRA=()
if [[ "${NO_CAPTURE:-0}" == 1 ]]; then EXTRA+=(--no-capture); fi
if [[ "${LOCAL_FILES_ONLY:-0}" == 1 ]]; then EXTRA+=(--local-files-only); fi
# Gemma access must already be authorized in Hugging Face (cached login or HF_TOKEN).
# Never print credentials. Existing project cache/login is reused.
python -u scripts/exploretom/evaluate_adversarial.py \
  --model "$MODEL" --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_ROOT/$MODEL" --dtype bfloat16 \
  --max-new-tokens 40 --max-input-tokens 4096 --resume ${EXTRA[@]+"${EXTRA[@]}"}
