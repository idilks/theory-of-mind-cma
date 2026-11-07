#!/bin/bash
#SBATCH --job-name=path_union41
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
#SBATCH --output=logs/path_union41_%A_%a.out
#SBATCH --error=logs/path_union41_%A_%a.err

# From the project root on Andes: sbatch slurm_scripts/path_patching.sh
# For the four-cell repair use path_patching_h200_repair.sh instead.
# Check first:                    sbatch --array=0 slurm_scripts/path_patching.sh
# 14 tasks = one per condition/template/direction cell (C1, C2: 3 templates x 2
# directions; K1: 1 x 2). Each task loads the model ONCE and measures all 41
# targets + 41 layer-matched controls in its cell:
#   L28-L39 senders -> all 26 saved L42-L46 receiver candidates;
#   L46 senders     -> all 40 L47 heads (EXPLORATORY; no known later receiver list).
# Final-token Q/K/V paths; shared native K/V groups measured once, aliases recorded.
# Frozen selection: results/aggregated/path_patching_setup/{plan.json,sender_control_map.csv}.
# 07:59 is a scheduler limit, NOT a runtime estimate; per-pair seconds are logged.
#
# Code snapshot: the first task copies the code + plan into
# $OUTPUT_ROOT/code_snapshot and EVERY task runs from that copy, so push.sh syncs
# during the run cannot change the study. To run changed code, use a new OUTPUT_ROOT.
# Resume: rerun the same sbatch; completed pairs are skipped, changed settings rejected.
# logs/ must exist in the project root before submission.

# Conda/MKL activation hooks may read unset variables. Keep error/pipe checks,
# but disable nounset until activation finishes (including when inherited via bash -u).
set -eo pipefail
set +u
PROJECT_DIR=${PROJECT_DIR:-/dartfs/rc/lab/F/FranklandS/tom}
PAIRS_PER_CELL=${PAIRS_PER_CELL:-10}
BATCH_SIZE=${BATCH_SIZE:-8}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_DIR/results/circuit/path_patching/union41_v2}
MODEL_CACHE=${MODEL_CACHE:-/dartfs/rc/lab/F/FranklandS/models/cache}
PLAN_REL=results/aggregated/path_patching_setup/plan.json
SNAPSHOT="$OUTPUT_ROOT/code_snapshot"

if (( $# > 0 )); then
    echo "This launcher runs the frozen full study without arguments. For custom runs use path_patching.py run." >&2
    exit 2
fi
: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch; this launcher requires an array task ID}"
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate "$PROJECT_DIR/envs/tom_analysis"
set -u
cd "$PROJECT_DIR"
echo "Launcher: union41_v2_per_cell; array task $SLURM_ARRAY_TASK_ID on $(hostname)"
export LD_LIBRARY_PATH="$PROJECT_DIR/envs/tom_analysis/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export PYTHONUNBUFFERED=1
python -c 'import sys, torch, transformer_lens, transformers, numpy; print("Environment:", sys.prefix); print("torch:", torch.__version__); assert torch.cuda.is_available(), "CUDA unavailable"'
python -c 'import sys, torch; sys.path.insert(0,"analysis_scripts/circuit"); from path_patching_h200_repair import validate_gpu_names; validate_gpu_names([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'

# Freeze the code once per study (first task wins; later tasks reuse it).
mkdir -p "$OUTPUT_ROOT"
(
    flock -x 9
    if [[ ! -d "$SNAPSHOT" ]]; then
        test -s "$PROJECT_DIR/$PLAN_REL" || { echo "Missing $PLAN_REL; sync results/aggregated/path_patching_setup first." >&2; exit 1; }
        staging=$(mktemp -d "$OUTPUT_ROOT/.code_snapshot.XXXXXX")
        mkdir -p "$staging/analysis_scripts/circuit" "$staging/$(dirname "$PLAN_REL")"
        rsync -a --exclude results --exclude wandb --exclude logs --exclude __pycache__ \
            --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' --exclude '*.out' --exclude '*.err' \
            "$PROJECT_DIR/cma_scripts" "$staging/"
        cp analysis_scripts/condition_registry.py "$staging/analysis_scripts/"
        cp analysis_scripts/circuit/path_patching.py analysis_scripts/circuit/path_patching_study.py "$staging/analysis_scripts/circuit/"
        cp "$PLAN_REL" "$staging/$PLAN_REL"
        mv "$staging" "$SNAPSHOT"
        echo "Created code snapshot $SNAPSHOT ($(du -sh "$SNAPSHOT" | cut -f1))"
    fi
) 9>"$OUTPUT_ROOT/.code_snapshot.lock"
PLAN="$SNAPSHOT/$PLAN_REL"

# Other jobs download through TRANSFORMERS_CACHE=$MODEL_CACHE (flat layout); a
# hub/ subdirectory is the HF_HOME default. Use whichever holds the pinned revision.
REVISION=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["revision"])' "$PLAN")
HUB_CACHE=""
for candidate in "$MODEL_CACHE" "$MODEL_CACHE/hub"; do
    if [[ -d "$candidate/models--Qwen--Qwen2.5-14B-Instruct/snapshots/$REVISION" ]]; then
        HUB_CACHE=$candidate
        break
    fi
done
if [[ -z "$HUB_CACHE" ]]; then
    echo "Pinned Qwen revision $REVISION not found under $MODEL_CACHE or $MODEL_CACHE/hub. Snapshots present:" >&2
    ls -d "$MODEL_CACHE"/models--Qwen--Qwen2.5-14B-Instruct/snapshots/* \
          "$MODEL_CACHE"/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/* >&2 2>/dev/null || true
    exit 1
fi
export HF_HOME=$MODEL_CACHE
export HF_HUB_CACHE=$HUB_CACHE
export TRANSFORMERS_CACHE=$HUB_CACHE
export HF_HUB_OFFLINE=1
echo "Model cache: $HUB_CACHE (revision $REVISION)"

python "$SNAPSHOT/analysis_scripts/circuit/path_patching_study.py" \
    --plan "$PLAN" --task-id "$SLURM_ARRAY_TASK_ID" \
    --pairs-per-cell "$PAIRS_PER_CELL" --batch-size "$BATCH_SIZE" --output-root "$OUTPUT_ROOT"
