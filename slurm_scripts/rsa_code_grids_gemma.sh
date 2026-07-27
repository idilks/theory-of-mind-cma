#!/bin/bash
#SBATCH --job-name=rsa_code_gemma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=03:59:00
#SBATCH --partition=h200_preemptable,preempt_wager
#SBATCH --array=0-1
#SBATCH --mem=100GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/rsa_code_gemma_%A_%a.out
#SBATCH --error=logs/rsa_code_gemma_%A_%a.err

# Full-head (46x32, GQA ungrouped) hook_z captures for the code RDM grids on
# Gemma-2-27B-it, with baseline-accuracy greedy decode.
# docs/design_code_pointer_rdm.md. Array 0 = code_position, 1 = code_ledger.
# Same stimuli as the Qwen jobs (answer pool = tokenizer intersection).

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

GRIDS=("code_position" "code_ledger")
GRID=${GRIDS[$SLURM_ARRAY_TASK_ID]}
echo "Job $SLURM_ARRAY_TASK_ID: grid=$GRID model=gemma-2-27b-it"

if [ ! -f results/rsa_head_outputs/stimuli_code_position/code_position.csv ]; then
  python scripts/rsa_head_outputs/build_code_position_stimuli.py || exit 1
fi
if [ ! -f results/rsa_head_outputs/stimuli_code_ledger/code_ledger.csv ]; then
  python scripts/rsa_head_outputs/build_code_ledger_stimuli.py || exit 1
fi

python scripts/rsa_head_outputs/capture_head_outputs_full.py \
  --template "$GRID" \
  --stim_dir  "results/rsa_head_outputs/stimuli_${GRID}" \
  --out_dir   "results/rsa_head_outputs/activations_${GRID}_full/gemma-2-27b-it" \
  --model_type gemma-2-27b-it \
  --ungroup_grouped_query_attention \
  --system_prompt code \
  --decode_answers
