#!/bin/bash
#SBATCH --job-name=circuit_s2_rr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=07:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --mem=90GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/circuit_s2_rr_%j.out
#SBATCH --error=logs/circuit_s2_rr_%j.err

# Reader-head restoration / transplantation under the Figure "steer-attn"
# residual edit (rank-1 direction, alpha = 4, resid_pre L42), on the same
# held-out prompts and 26 readers as circuit_stage2_steering.sh. Per
# prompt: clean, steered, and for each of q/k/v/pattern a restore (steer
# on, readers' activation <- clean), a transplant (steer off, readers'
# activation <- steered) and three layer-matched random-head control
# restores. 22 forward passes per prompt x 450 prompts. Forward passes only;
# rows are appended per completed prompt and finished prompts are skipped
# on restart, so preemption is recoverable — just resubmit.
#
# Pilot first (5 prompts per variant, separate run label), then the full
# run; the pilot's identity.csv must show max_abs_logit_diff ~ 0 for every
# site before the full run is trusted.
#
# Prerequisites on the HPC tree: the same as circuit_stage2_steering.sh
# (stage-1 capture + weights.pt; directions_L41.npz synced via
# sync_to_hpc.sh).

export LD_LIBRARY_PATH=/dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis/lib:$LD_LIBRARY_PATH
export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd
mkdir -p .conda/pkgs/cache .conda/envs

cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

if [ ! -f results/aggregated/circuit/stage1/directions_L41.npz ]; then
  echo "ERROR: stage-1 directions missing — run stage1_geometric_screen.py locally and sync." >&2
  exit 1
fi

echo "Part A: pilot (5 prompts per variant, identity checks on all of them)"
python scripts/circuit/stage2_reader_restoration.py \
  --site L41 --restore q k v pattern \
  --limit 5 --n_identity 5 \
  --run_label reader_restore_pilot || exit 1

python - <<'PY' || exit 1
import pandas as pd
d = pd.read_csv("results/circuit/stage2_steering/Qwen2.5-14B-Instruct/L41_reader_restore_pilot/identity.csv")
print(d.groupby(["site", "source"])["max_abs_logit_diff"].max())
worst = d["max_abs_logit_diff"].max()
print(f"worst identity |logit diff| = {worst:.3e}")
if worst > 1e-2:
    raise SystemExit("identity check failed: restoration from a run's own activations does not reproduce it")
PY

echo "Part B: full run (225 held-out prompts per variant)"
python scripts/circuit/stage2_reader_restoration.py \
  --site L41 --restore q k v pattern \
  --run_label reader_restore
