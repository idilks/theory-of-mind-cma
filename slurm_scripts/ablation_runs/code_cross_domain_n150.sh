#!/bin/bash
#SBATCH --job-name=cross_n150
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=07:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --mem=90GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/cross_n150_%A_%a.out
#SBATCH --error=logs/cross_n150_%A_%a.err

# Cross-DOMAIN ablation transfer at n=150, both directions, primary protocol
# (matched belief system prompt + per-variant answer prefill).
#
# WHY: the cross-domain transfer is the centrepiece of the rebuttal to R2's
# memorized-event-structure objection, and it is currently the thinnest
# evidence in the paper by sample size:
#
#   code heads -> ToM   n=48   (code_cross_code_abstract_on_tom_matched_prefill,
#                               launched with --prompt_num 50)
#   ToM heads -> code   n=75   (tom_c1_full1920_on_code_tom_*, the Yang-style
#                               full-1920 sweep, deliberately n=75 for cost)
#   ToM heads -> code   n=150  LAUNCHED BUT NEVER COMPLETED — the directory
#                               results/ablation/tom_c1_on_code_prefill_*
#                               contains head_list.csv and the prompt files but
#                               no cumulative_ablation_results.csv.
#
# Note the cross-FRAME (belief<->photo) transfer IS at n=150 already, but only
# under the ORIGINAL protocol (belief_heads/, photo_heads/, neutral prompt,
# free generation, false_belief baseline 0.633). Every matched-prompt+prefill
# ToM run in the tree is n=48, because they all pass --prompt_num 50. If the
# n=48 cells are being quoted anywhere in the paper, this job fixes them too.
#
#   task 0  code_abstract heads (12) -> ToM evals at n=150
#           Same config as code_cross_code_heads_on_tom_matched_prefill.sh
#           with --prompt_num 50 -> 150. Writes to a NEW directory so the
#           existing n=48 result stays available for comparison.
#   task 1  T_C1 heads (32) -> code evals at n=150
#           Completes the run that died. Uses the published T_C1 set rather
#           than a fingerprint top-k, so it is directly comparable to the
#           within-ToM ablation table.
#   task 2  T_C5 heads (22) -> code evals at n=150
#           The magnitude comparator for task 1, free from the same CSV.
#           NOT a clean specificity control, and do not present it as one:
#           T_C5 shares 11 of its 22 heads with T_C1 (32/22, 11 shared, 21
#           C1-only, 11 C5-only), so it will flip code_copy through the shared
#           core. The full-1920 sweep already shows exactly this — the
#           C5-ordering flip on code_copy turns on (0.006 at k=8 -> 0.169 at
#           k=16) precisely where the C1 core enters the C5 ranking. What this
#           task buys is a same-protocol, same-n magnitude comparison at the
#           two published set sizes, which the table in the rebuttal wants.
#           The clean specificity test is set subtraction (a_only / b_only /
#           intersection) — that is decomp_c1c5_on_code.sh, which is written
#           but has never run. Submit that too if there is GPU budget.
#
# Runtime estimate. The one documented rate for this script is ~4.85 min per
# (k-step x curve) at 375 evals (decomp_c1c5_on_code.sh; the same rate that
# gave ~39 h for the 241-step full-1920 sweep), i.e. ~0.78 s/eval. Each task is
# (|set| + 1) k-steps x 2 curves (target + layer-matched control) x n_variants
# x 150 prompts:
#   task 0   13 steps x 2 x 3 x 150 = 11,700 evals ~ 2.5 h
#   task 1   33 steps x 2 x 2 x 150 = 19,800 evals ~ 4.3 h
#   task 2   23 steps x 2 x 2 x 150 = 13,800 evals ~ 3.0 h
# All fit the 07:59 wall time with margin. Treat these as estimates: the rate
# was measured on ToM prompts at max_new_tokens 5, and the code prompts are
# longer and run at 8. ablation_experiment.py checkpoints after every k step
# and resumes, and --requeue is set, so a preemption costs one step, not a job.
# If the queue is slow, drop --prompt_num to 75 to roughly halve all three.
#
# All three tasks are independent.

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
mkdir -p logs

CODE_CSV=results/aggregated/code_head_sets/code_heads_end.csv
TOM_CSV=results/aggregated/code_head_sets/tom_c1_c5_heads.csv
for f in $CODE_CSV $TOM_CSV; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not present on HPC — build locally and sync." >&2
    exit 1
  fi
done

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  echo "Task 0: code_abstract heads -> ToM evals, n=150, matched+prefill"
  python cma_scripts/tasks/causal_analysis/ablation_experiment.py \
    --model_type Qwen2.5-14B-Instruct \
    --prompt_num 150 \
    --max_new_tokens 5 \
    --templates food_truck hair_styling library_book \
    --seed 1 \
    --n_devices 1 \
    --device_map cuda \
    --head_source per_condition \
    --head_csv $CODE_CSV \
    --condition code_abstract \
    --rank_by mean_score \
    --eval_variants false_belief true_belief basic_ab \
    --system_prompt "You are a helpful assistant reasoning about agents' beliefs." \
    --assistant_prefill per_variant \
    --run_label code_abstract_on_tom_matched_prefill_n150 \
    --output_dir results/ablation/code_cross_code_abstract_on_tom_matched_prefill_n150
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  echo "Task 1: T_C1 heads -> code evals, n=150, prefill per_variant"
  python cma_scripts/tasks/causal_analysis/ablation_experiment.py \
    --model_type Qwen2.5-14B-Instruct \
    --prompt_num 150 \
    --max_new_tokens 8 \
    --seed 1 \
    --n_devices 1 \
    --device_map cuda \
    --head_source per_condition \
    --head_csv $TOM_CSV \
    --condition tom_c1 \
    --rank_by mean_score \
    --eval_variants code_copy code_alias \
    --assistant_prefill per_variant \
    --run_label tom_c1_on_code_prefill_per_variant_n150 \
    --output_dir results/ablation/tom_c1_on_code_prefill_per_variant_n150
else
  echo "Task 2: T_C5 heads -> code evals, n=150, prefill per_variant"
  python cma_scripts/tasks/causal_analysis/ablation_experiment.py \
    --model_type Qwen2.5-14B-Instruct \
    --prompt_num 150 \
    --max_new_tokens 8 \
    --seed 1 \
    --n_devices 1 \
    --device_map cuda \
    --head_source per_condition \
    --head_csv $TOM_CSV \
    --condition tom_c5 \
    --rank_by mean_score \
    --eval_variants code_copy code_alias \
    --assistant_prefill per_variant \
    --run_label tom_c5_on_code_prefill_per_variant_n150 \
    --output_dir results/ablation/tom_c5_on_code_prefill_per_variant_n150
fi
