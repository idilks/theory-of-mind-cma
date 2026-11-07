#!/bin/bash
#SBATCH --job-name=steer_photo_on_code
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager,preempt_lsong
#SBATCH --mem=120GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/steer_photo_on_code_%j.out
#SBATCH --error=logs/steer_photo_on_code_%j.err

# Photo axis on code prompts, the missing cell of paper Fig. 5c. Saves only at
# the end, so a preempted or timed-out job must be resubmitted from scratch.
#
# Outputs: results/rsa_head_outputs/steering_code_photo_qwen/
# Success: the log's last lines say "all exact = True" and "OK".

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

python analysis_scripts/rsa_head_outputs/steer_photo_axis_on_code.py --device cuda
