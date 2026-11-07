#!/bin/bash -l
#SBATCH --job-name=exploretom_div
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=h200_preemptable,preempt_wager
#SBATCH --mem=100GB
#SBATCH --time=05:59:00
#SBATCH --output=logs/exploretom_div_%j.out
#SBATCH --error=logs/exploretom_div_%j.err

# Submit from project root after push.sh has synced scripts/ and slurm_scripts/.
# Walltime is an allocation limit, not a runtime estimate. Qwen only.
set -eo pipefail
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
PROJECT_ROOT="${PROJECT_ROOT:-/dartfs/rc/lab/F/FranklandS/tom}"
conda activate "${TOM_ENV:-$PROJECT_ROOT/envs/tom_analysis}"
set -u
cd "$PROJECT_ROOT"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
RUN_DIR="${EXPLORETOM_RUN_DIR:-results/exploretom/released_100_seed0/qwen}"
REFERENCE="${EXPLORETOM_REFERENCE:-scripts/exploretom/data/divergence_reference_qwen}"
OUTPUT="${EXPLORETOM_ANALYSIS_OUTPUT:-results/exploretom/divergence_qwen_${SLURM_JOB_ID:?Submit using sbatch}}"
test -s "$REFERENCE/manifest.json"
test -s "$RUN_DIR/run_manifest.json"
EXTRA=()
if [[ -n "${EXPLORETOM_REUSE_REFERENCE:-}" ]]; then
  EXTRA+=(--reuse-reference-from "$EXPLORETOM_REUSE_REFERENCE")
fi
# Authorized exclusion after job 9934744: keep the 21 validated original heads.
python -u scripts/exploretom/analyze_divergence.py run \
  --reference "$REFERENCE" --run-dir "$RUN_DIR" --output "$OUTPUT" \
  --exclude-head 36 30 ${EXTRA[@]+"${EXTRA[@]}"}
