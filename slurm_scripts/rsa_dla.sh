#!/bin/bash
#SBATCH --job-name=rsa_dla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager,preempt_lsong
#SBATCH --array=0-2%3
#SBATCH --mem=100GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/rsa_dla_%A_%a.out
#SBATCH --error=logs/rsa_dla_%A_%a.err

# Direct logit attribution for every (layer, head) on the balanced RSA
# stimulus set. Reads z from activations_full/{template}.pt, projects through
# W_O and the unembedding for the answer + distractor first tokens, writes
# a long-format CSV.
#
# Outputs:
#   results/rsa_head_outputs/dla/{template}_dla.csv

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

TEMPLATES=("food_truck" "hair_styling" "library_book")
TEMPLATE=${TEMPLATES[$SLURM_ARRAY_TASK_ID]}

echo "Job $SLURM_ARRAY_TASK_ID: template=$TEMPLATE"

python analysis_scripts/rsa_head_outputs/compute_dla.py --template $TEMPLATE

echo "done. Output: results/rsa_head_outputs/dla/${TEMPLATE}_dla.csv"
