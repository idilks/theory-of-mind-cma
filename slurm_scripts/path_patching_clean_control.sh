#!/bin/bash
#SBATCH --job-name=path_clean41
#SBATCH --array=0-13%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=07:59:00
#SBATCH --partition=h200_preemptable,preempt_wager
#SBATCH --mem=90GB
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/path_clean41_%A_%a.out
#SBATCH --error=logs/path_clean41_%A_%a.err

# Run from the Andes project root:
#   sbatch slurm_scripts/path_patching_clean_control.sh
# For the four-cell repair use path_patching_clean_h200_repair.sh instead.
# Replays ALL 14 cells, original 140 prompts, both receiver scopes, all Q/K/V
# routes for 41 selected heads + 41 layer-matched controls. Identical clean
# batches are measured once and mapped to every original sender/path row.
# Uses the ORIGINAL executed hooks from SOURCE_ROOT/code_snapshot, frozen
# prompts and exact batch widths/order. No donor patch, no prompt generation.
# Numerical failures are saved, not discarded. Read each cell's summary.json.
# Originals are read-only; outputs go to a separate study. Resume with same sbatch.
# Dependencies must match original manifests. Both flat and hub/ model caches
# are supported; offline mode prevents accidental model downloads/revisions.
# 07:59 is a scheduler limit, not a runtime estimate. logs/ must exist at submission.
set -eo pipefail
set +u
PROJECT_DIR=${PROJECT_DIR:-/dartfs/rc/lab/F/FranklandS/tom}
SOURCE_ROOT=${SOURCE_ROOT:-$PROJECT_DIR/results/circuit/path_patching/union41_v2}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_DIR/results/circuit/path_patching/union41_v2_clean_control_v1}
MODEL_CACHE=${MODEL_CACHE:-/dartfs/rc/lab/F/FranklandS/models/cache}
SNAPSHOT="$OUTPUT_ROOT/code_snapshot"
if (( $# > 0 )); then
    echo "Submit without arguments; settings and all 14 cells are included." >&2
    exit 2
fi
: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch; array task ID required}"
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate "$PROJECT_DIR/envs/tom_analysis"
set -u
cd "$PROJECT_DIR"
export LD_LIBRARY_PATH="$PROJECT_DIR/envs/tom_analysis/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export PYTHONUNBUFFERED=1
python -c 'import sys, torch, transformer_lens, transformers; print("Environment:", sys.prefix, torch.__version__); assert torch.cuda.is_available(), "CUDA unavailable"'
python -c 'import sys, torch; sys.path.insert(0,"analysis_scripts/circuit"); from path_patching_h200_repair import validate_gpu_names; validate_gpu_names([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
# Reject overlapping paths before creating any files.
python -c 'from pathlib import Path; import sys; a,b=map(lambda p:Path(p).resolve(),sys.argv[1:]); assert a!=b and a not in b.parents and b not in a.parents, "Source/output roots must be separate"' "$SOURCE_ROOT" "$OUTPUT_ROOT"
test -s "$SOURCE_ROOT/study.json"
mkdir -p "$OUTPUT_ROOT"
(
    flock -x 9
    if [[ ! -d "$SNAPSHOT" ]]; then
        staging=$(mktemp -d "$OUTPUT_ROOT/.code_snapshot.XXXXXX")
        mkdir -p "$staging/analysis_scripts/circuit"
        cp analysis_scripts/circuit/path_patching_clean_control.py "$staging/analysis_scripts/circuit/"
        cp slurm_scripts/path_patching_clean_control.sh "$staging/launcher.sh"
        mv "$staging" "$SNAPSHOT"
    fi
) 9>"$OUTPUT_ROOT/.code_snapshot.lock"
REVISION=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan"]["revision"])' "$SOURCE_ROOT/study.json")
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
export HF_HUB_OFFLINE=1
printf 'Clean control task %s; source=%s; output=%s; cache=%s\n' "$SLURM_ARRAY_TASK_ID" "$SOURCE_ROOT" "$OUTPUT_ROOT" "$HUB_CACHE"
python "$SNAPSHOT/analysis_scripts/circuit/path_patching_clean_control.py" \
    --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" --task-id "$SLURM_ARRAY_TASK_ID"
