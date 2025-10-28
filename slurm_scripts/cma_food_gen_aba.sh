#!/bin/bash
#SBATCH --job-name=cma_gen_lib        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=1000GB         # total memory
#SBATCH --gres=gpu:1           # number of gpus per node
#SBATCH --time=3:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slurm_logs/cma_library%j.out
#SBATCH --error=slurm_logs/cma_library_%j.err
#SBATCH --partition=h200_preemptable
#SBATCH --mail-type=BEGIN,END,FAIL

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

# python codebase/tasks/identity_rules/cma.py --use_behavioral_tom --context_type  abstract --base_rule ABA --template_names food_truck --prompt_num 500 --max_new_tokens 17 --samples_per_condition 1 --activation_name z
# # python codebase/tasks/identity_rules/cma.py --use_behavioral_tom --context_type  abstract --base_rule ABA --template_names library_book --prompt_num 500 --max_new_tokens 10 --samples_per_condition 1 --activation_name z
# # python codebase/tasks/identity_rules/cma.py --use_behavioral_tom --context_type  abstract --base_rule ABA --template_names detailed_object_ --prompt_num 500 --max_new_tokens 10 --samples_per_condition 1 --activation_name z

#  python codebase/tasks/identity_rules/cma.py  --use_behavioral_tom --context_type abstract --base_rule ABA --template_names food_truck --prompt_num 20 --max_new_tokens 20 --activation_name z --model_type Qwen2.5-32B --question_style completion  


# lets try this again but now on resid_post
# python codebase/tasks/identity_rules/cma.py  --use_behavioral_tom --context_type abstract --base_rule ABA --template_names food_truck --prompt_num 20 --max_new_tokens 15 --activation_name resid_post --model_type Qwen2.5-32B --question_style completion  --samples_per_condition 1 

# python codebase/tasks/identity_rules/cma.py  --use_behavioral_tom --context_type abstract --base_rule ABA --template_names food_truck --prompt_num 20 --max_new_tokens 15 --activation_name resid_post --model_type Qwen2.5-32B --question_style generation  --samples_per_condition 1 

python codebase/tasks/identity_rules/cma.py  --use_behavioral_tom --context_type abstract --base_rule ABA --template_names food_truck --prompt_num 50 --max_new_tokens 10 --activation_name z --model_type Qwen2.5-14B-Instruct --question_style instruction