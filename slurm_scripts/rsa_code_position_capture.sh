#!/bin/bash
#SBATCH --job-name=rsa_code_pos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --mem=80GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/rsa_code_pos_%j.out
#SBATCH --error=logs/rsa_code_pos_%j.err

# Capture per-head hook_z outputs across the code-position stimulus grid
# (pointer RDM test, week plan Step 6). 144 stimuli, forward pass only —
# no patching — so one job covers the whole set.
# Heads: union of code_abstract + code_control + T_C1 + T_C5.

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

# Build stimuli + union head CSV if absent. The builder runs its design-time
# RDM separability check and exits nonzero if the grid is confounded, so a
# bad design never reaches the GPU.
if [ ! -f results/rsa_head_outputs/stimuli_code_position/code_position.csv ]; then
  echo "Building code-position stimulus grid..."
  python analysis_scripts/rsa_head_outputs/build_code_position_stimuli.py || exit 1
fi

python analysis_scripts/rsa_head_outputs/capture_head_outputs.py \
  --template code_position \
  --stim_dir  results/rsa_head_outputs/stimuli_code_position \
  --heads_csv results/rsa_head_outputs/topk_heads/code_position_union.csv \
  --out_dir   results/rsa_head_outputs/activations_code_position \
  --system_prompt code
