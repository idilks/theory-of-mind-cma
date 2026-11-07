#!/bin/bash
#SBATCH --job-name=head_resid_bridge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=90GB
#SBATCH --time=07:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager,preempt_lsong
#SBATCH --output=head_resid_bridge_%j.out
#SBATCH --error=head_resid_bridge_%j.err

# Submit from the project root: sbatch slurm_scripts/head_residual_bridge.sh
# Full held-out evaluation: sbatch --export=ALL,PHASE=full slurm_scripts/head_residual_bridge.sh
# The time limit is a queue cap, not an estimated runtime. Measure the pilot first.
# The runner checkpoints prompts and permits resume only with identical inputs/code.
set -eo pipefail
set +u
PROJECT_DIR=${PROJECT_DIR:-/dartfs/rc/lab/F/FranklandS/tom}
MODEL_CACHE=${MODEL_CACHE:-/dartfs/rc/lab/F/FranklandS/models/cache}
PHASE=${PHASE:-pilot}
if (( $# > 0 )); then
    echo "Submit without arguments; use PHASE=pilot or PHASE=full." >&2
    exit 2
fi
case "$PHASE" in pilot|full) ;; *) echo "PHASE must be pilot or full" >&2; exit 2;; esac
cd "$PROJECT_DIR"
PROJECT_DIR=$(pwd -P)
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate "$PROJECT_DIR/envs/tom_analysis"
set -u
test "${CONDA_PREFIX:-}" = "$PROJECT_DIR/envs/tom_analysis" || {
    echo "Expected activated environment $PROJECT_DIR/envs/tom_analysis; got ${CONDA_PREFIX:-unset}" >&2
    exit 1
}
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REVISION=cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
HUB_CACHE=""
for candidate in "$MODEL_CACHE" "$MODEL_CACHE/hub"; do
    if [[ -d "$candidate/models--Qwen--Qwen2.5-14B-Instruct/snapshots/$REVISION" ]]; then
        HUB_CACHE=$candidate
        break
    fi
done
if [[ -z "$HUB_CACHE" ]]; then
    echo "Pinned Qwen revision $REVISION missing from $MODEL_CACHE and $MODEL_CACHE/hub" >&2
    exit 1
fi
export HF_HOME=$MODEL_CACHE
export HF_HUB_CACHE=$HUB_CACHE
export TRANSFORMERS_CACHE=$HUB_CACHE

RUNNER=analysis_scripts/circuit/stage2_steering.py
for required in "$RUNNER" \
    results/aggregated/circuit/stage1/directions_L41.npz \
    results/rsa_head_outputs/activations_code_position_full/Qwen2.5-14B-Instruct/code_position.pt \
    results/circuit/stage1_capture/Qwen2.5-14B-Instruct/weights.pt \
    results/circuit/stage1_capture/Qwen2.5-14B-Instruct/food_truck.pt \
    results/circuit/stage1_capture/Qwen2.5-14B-Instruct/hair_styling.pt \
    results/circuit/stage1_capture/Qwen2.5-14B-Instruct/library_book.pt; do
    test -s "$required" || { echo "Missing or empty required file: $required" >&2; exit 1; }
done

python -c 'import sys, torch, numpy, pandas, matplotlib, transformers, transformer_lens; from importlib.metadata import version; print("Environment:", sys.executable, sys.prefix); print("Versions:", {name: version(name) for name in ("torch", "numpy", "pandas", "matplotlib", "transformers", "transformer-lens")})'
args=(--head_residual_bridge --bridge_phase "$PHASE" --bridge_pairs 2
    --bridge_seed 20261023 --bridge_revision "$REVISION" --bridge_dtype bfloat16
    --bridge_output "$PROJECT_DIR/results/circuit/head_residual_bridge/v1_${PHASE}")
# Validate capture metadata, fit/evaluation separation, and inputs before loading weights.
python "$RUNNER" "${args[@]}" --bridge_preflight
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; print("Visible GPUs:", names); assert names and all(any(kind in name.upper() for kind in ("H200", "A100")) for name in names), f"A100 or H200 required for every visible GPU; got {names}"'
python "$RUNNER" "${args[@]}"
