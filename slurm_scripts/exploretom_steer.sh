#!/bin/bash -l
#SBATCH --job-name=exploretom_steer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager,preempt_lsong
#SBATCH --mem=100GB
#SBATCH --time=03:59:00
#SBATCH --output=logs/exploretom_steer_%j.out
#SBATCH --error=logs/exploretom_steer_%j.err

# Experiment A: steer divergence heads along the original state axis on the
# 56 ExploreToM location questions; matched random-direction control.
# Submit from the project root after push.sh has synced scripts/ and slurm_scripts/.
# Needs the completed divergence analysis dir and the original example JSONs
# (results/exploretom/released_100_seed0/qwen/examples) on the cluster.
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
ANALYSIS="${EXPLORETOM_ANALYSIS_DIR:-results/exploretom/divergence_qwen_9976575}"
EXAMPLES="${EXPLORETOM_EXAMPLES_DIR:-results/exploretom/released_100_seed0/qwen/examples}"
# The hand review lives under scripts/ (synced by push.sh); the copy under
# results/exploretom/released_100_seed0/answer_review/ exists only locally.
REVIEW="${EXPLORETOM_REVIEW:-scripts/exploretom/data/divergence_reference_qwen/review.jsonl}"
OUTPUT="${EXPLORETOM_STEER_OUTPUT:-results/exploretom/steer_qwen_${SLURM_JOB_ID:?Submit using sbatch}}"
for f in "$ANALYSIS/reference_replay.npz" "$ANALYSIS/exploretom_z.npz" "$REVIEW"; do
  [[ -s "$f" ]] || { echo "Missing input: $f" >&2; exit 1; }
done
[[ -d "$EXAMPLES" ]] || { echo "Missing examples dir: $EXAMPLES" >&2; exit 1; }
python -u scripts/exploretom/steer_failures.py \
  --analysis-dir "$ANALYSIS" --examples-dir "$EXAMPLES" --require-examples --review "$REVIEW" \
  --output "$OUTPUT" --exclude-head 36 30 --alphas -1 -0.5 0.5 1 --n-random 5 \
  --also-all-included --local-files-only "$@"
