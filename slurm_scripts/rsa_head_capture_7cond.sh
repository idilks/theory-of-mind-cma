#!/bin/bash
#SBATCH --job-name=rsa_head_capture_7cond
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager,preempt_lsong
#SBATCH --array=0-2%6
#SBATCH --mem=80GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/rsa_head_capture_7cond_%A_%a.out
#SBATCH --error=logs/rsa_head_capture_7cond_%A_%a.err

# Capture per-head hook_z outputs for the union of top-20 heads across all 7
# canonical conditions (~50-70 unique heads, vs 34 in the C1+C5-only run).
# Lifts the C1/C5 selection bias so per-head RDM analyses (Mantel C1<->C2,
# per-head belief_effect scatter across conditions) are not constrained to
# heads selected for the C1/C5 dissociation question.
#
# Outputs to a parallel directory so the 34-head capture stays intact:
#   results/rsa_head_outputs/topk_heads_7cond/union_top20.csv
#   results/rsa_head_outputs/activations_7cond/{template}.pt

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

CONDITIONS=(
  "C1_abstract_belief"
  "C2_abstract_photo"
  "C3_answer_changes_belief"
  "C4_answer_changes_photo"
  "C5_answer_changes_location_only"
  "C6_answer_changes_location_only_photo"
  "C7_state_only_no_belief"
)

TOPK_DIR=results/rsa_head_outputs/topk_heads_7cond
UNION_CSV=${TOPK_DIR}/union_top20.csv
ACT_DIR=results/rsa_head_outputs/activations_7cond

echo "Job $SLURM_ARRAY_TASK_ID: template=$TEMPLATE"

if [ ! -f results/rsa_head_outputs/stimuli/${TEMPLATE}.csv ]; then
  echo "Building stimulus set..."
  python analysis_scripts/rsa_head_outputs/build_stimulus_set.py
fi

if [ ! -f $UNION_CSV ]; then
  echo "Selecting top-20 heads across 7 conditions..."
  python analysis_scripts/rsa_head_outputs/select_topk_heads.py \
    --k 20 \
    --conditions "${CONDITIONS[@]}" \
    --out_dir $TOPK_DIR
  echo "Union size:"
  wc -l $UNION_CSV
fi

mkdir -p $ACT_DIR
python analysis_scripts/rsa_head_outputs/capture_head_outputs.py \
  --template $TEMPLATE \
  --heads_csv $UNION_CSV \
  --stim_dir  results/rsa_head_outputs/stimuli \
  --out_dir   $ACT_DIR
