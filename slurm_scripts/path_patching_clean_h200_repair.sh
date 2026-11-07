#!/bin/bash
#SBATCH --job-name=clean_h200_fix
#SBATCH --array=2,5,8,13%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=07:59:00
#SBATCH --partition=h200_preemptable,preempt_wager
#SBATCH --mem=90GB
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/clean_h200_fix_%A_%a.out
#SBATCH --error=logs/clean_h200_fix_%A_%a.err

# Submit from /dartfs/rc/lab/F/FranklandS/tom, after scripts have synced:
#   sbatch slurm_scripts/path_patching_clean_h200_repair.sh
# control repair ONLY: tasks 2,5,8,13. All 41 targets + 41 controls in each cell.
# Exact original prompts, model revision, dtype, batches and executed code.
# Original studies and the other ten cells are never overwritten or copied as results.
# Repeating this command resumes incomplete repair checkpoints.
# Scheduler restricted to H200; runtime also rejects non-H200 allocations.
# 07:59 is a scheduler limit, not a measured runtime estimate.
set -eo pipefail
set +u
PROJECT_DIR=${PROJECT_DIR:-/dartfs/rc/lab/F/FranklandS/tom}
SOURCE_ROOT=${SOURCE_ROOT:-$PROJECT_DIR/results/circuit/path_patching/union41_v2}
CONTROL_SOURCE_ROOT=${CONTROL_SOURCE_ROOT:-$PROJECT_DIR/results/circuit/path_patching/union41_v2_clean_control_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_DIR/results/circuit/path_patching/union41_v2_h200_clean_repair_v1}
MODEL_CACHE=${MODEL_CACHE:-/dartfs/rc/lab/F/FranklandS/models/cache}
if (( $# > 0 )); then echo "Submit without arguments; the four repair tasks are built in." >&2; exit 2; fi
: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch; array task ID required}"
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate "$PROJECT_DIR/envs/tom_analysis"
set -u
cd "$PROJECT_DIR"
export LD_LIBRARY_PATH="$PROJECT_DIR/envs/tom_analysis/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_DIR/analysis_scripts/circuit${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch; from path_patching_h200_repair import validate_gpu_names; names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; validate_gpu_names(names); print("GPUs:", names)'
# Validate both protected roots and task scope BEFORE any output writes.
python -c 'import sys; from pathlib import Path; from path_patching_h200_repair import validate_output, validate_task; validate_output(*map(Path,sys.argv[1:4])); validate_task("control",int(sys.argv[4]))' "$OUTPUT_ROOT" "$SOURCE_ROOT" "$CONTROL_SOURCE_ROOT" "$SLURM_ARRAY_TASK_ID"
REVISION=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan"]["revision"])' "$SOURCE_ROOT/study.json")
HUB_CACHE=""
for candidate in "$MODEL_CACHE" "$MODEL_CACHE/hub"; do
    if [[ -d "$candidate/models--Qwen--Qwen2.5-14B-Instruct/snapshots/$REVISION" ]]; then
        HUB_CACHE=$candidate
        break
    fi
done
if [[ -z "$HUB_CACHE" ]]; then echo "Pinned revision $REVISION missing from $MODEL_CACHE and $MODEL_CACHE/hub" >&2; exit 1; fi
export HF_HOME=$MODEL_CACHE
export HF_HUB_CACHE=$HUB_CACHE
export TRANSFORMERS_CACHE=$HUB_CACHE
export HF_HUB_OFFLINE=1
# Freeze this repair wrapper independently of the original executed assay snapshot.
mkdir -p "$OUTPUT_ROOT"
SNAPSHOT="$OUTPUT_ROOT/repair_driver_snapshot"
(
    flock -x 9
    if [[ ! -d "$SNAPSHOT" ]]; then
        staging=$(mktemp -d "$OUTPUT_ROOT/.repair_driver.XXXXXX")
        cp analysis_scripts/circuit/path_patching_h200_repair.py "$staging/"
        cp "$0" "$staging/launcher.sh"
        mv "$staging" "$SNAPSHOT"
    fi
) 9>"$OUTPUT_ROOT/.repair_driver.lock"
printf 'H200 control repair task %s on %s; output=%s\n' "$SLURM_ARRAY_TASK_ID" "$(hostname)" "$OUTPUT_ROOT"
python "$SNAPSHOT/path_patching_h200_repair.py" --kind control --task-id "$SLURM_ARRAY_TASK_ID" \
    --source-root "$SOURCE_ROOT" --control-source-root "$CONTROL_SOURCE_ROOT" --output-root "$OUTPUT_ROOT"
