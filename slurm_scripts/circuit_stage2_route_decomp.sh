#!/bin/bash
#SBATCH --job-name=circuit_s2_route
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=07:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --mem=90GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/circuit_s2_route_%j.out
#SBATCH --error=logs/circuit_s2_route_%j.err


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

# Three-route accounting of the Figure "steer-attn" residual edit
# (analysis_scripts/circuit/stage2_route_decomposition.py): pin direct / all
# attention outputs / all MLP outputs of L42-47 at the final token to
# clean or steered (8 exact configurations -> Shapley shares), plus the
# all-heads attention-weight route and per-layer necessity. 22 forward
# passes per prompt x 450 prompts, same prompts and edit as
# circuit_stage2_reader_restore.sh. Rows append per completed prompt;
# finished prompts are skipped on restart, so just resubmit if preempted.

echo "Part A: pilot (5 prompts per variant) + exactness check"
python analysis_scripts/circuit/stage2_route_decomposition.py \
  --site L41 --limit 5 --run_label route_decomp_pilot || exit 1

python - <<'PY' || exit 1
import pandas as pd
d = pd.read_csv("results/circuit/stage2_steering/Qwen2.5-14B-Instruct/L41_route_decomp_pilot/exact.csv")
print(d[["delta_norm", "resid_final_shift_norm", "scc_max_abs_err", "scc_rel_err"]].describe())
if d["scc_rel_err"].max() > 1e-2:
    raise SystemExit("pinned configuration does not reproduce clean + delta; stop")
PY

echo "Part B: full run"
python analysis_scripts/circuit/stage2_route_decomposition.py \
  --site L41 --run_label route_decomp
