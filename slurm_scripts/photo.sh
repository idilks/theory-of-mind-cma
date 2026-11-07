#!/bin/bash
#SBATCH --job-name=abstract_cma   # create a short name for your job
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks=1                # total number of tasks across all nodes
#SBATCH --cpus-per-task=1         # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1              # number of gpus per node
#SBATCH --time=10:59:00           # total run time limit (HH:MM:SS)
#SBATCH --partition=h200_preemptable
#SBATCH --mem=200GB               # total memory
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-3
#SBATCH --output=logs/photo_cma_%A_%a.out
#SBATCH --error=logs/photo_cma_%A_%a.err

export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd
mkdir -p .conda/pkgs/cache .conda/envs

cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

# Arrays for different experimental conditions (all templates × both rules × both patch conditions)
TEMPLATES=("food_truck" "library_book" "hair_styling" "basic_object_move_detailed")

python cma_scripts/tasks/causal_analysis/cma.py \
  --use_template_system \
  --context_type photo \
  --base_rule ABA \
  --template_names ${TEMPLATES[$SLURM_ARRAY_TASK_ID]} \
  --prompt_num 50 \
  --max_new_tokens 5 \
  --activation_name z \
  --model_type Qwen2.5-14B-Instruct \
  --question_style instruction \
  --run_statistical_tests \
  --n_permutations 1000 