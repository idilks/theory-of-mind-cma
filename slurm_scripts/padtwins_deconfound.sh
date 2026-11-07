#!/bin/bash
#SBATCH --job-name=padtwins_deconf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=07:59:00
#SBATCH --partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager
#SBATCH --array=0-3
#SBATCH --mem=110GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/padtwins_deconf_%A_%a.out
#SBATCH --error=logs/padtwins_deconf_%A_%a.err

# Deconfounding the pad response: is it positional INHERITANCE (a head's
# output mechanically carries the position of whatever it attends to, via
# RoPE on the value vectors) or a real pointer? The random-head baseline
# (already run, CPU) showed the pad response is generic — arbitrary heads
# show ~95% of the CMA sets'. Two questions remain, and these four jobs
# answer them.
#
#   tasks 0,1  ATTENTION-SUMMARY CAPTURE (unblocks deconfound step 1).
#              Rerun the 864-stimulus pad-twin capture with reduced
#              attention readouts on: per (stimulus, layer, head), the
#              attention-weighted mean source position, entropy, and mass on
#              the queried-slot / mutated / code spans at the final token.
#              Raw patterns are NOT stored (they would be hundreds of GB);
#              the mean source position is the only quantity the inheritance
#              control needs. NOTE: an earlier writeup comment claimed "we
#              have the patterns" — we did not. The code captures are hook_z
#              only, which is exactly why this step was blocked.
#              Output lands in a SEPARATE activations_code_padtwins_attn/
#              tree so the published Exp 3 capture is not overwritten.
#              Then, on the laptop:
#                python analysis_scripts/rsa_head_outputs/padtwins_position_prediction.py
#              R^2 ~ 1 from mean source position, and pad response gone after
#              partialling it out, means pure inheritance.
#
#   tasks 2,3  CAUSAL PAD-SUBSPACE ABLATION (deconfound step 3 + the
#              projection section's missing causal test). Projects the k=2
#              pad subspace out of each divergence head's z at the final
#              token in a live forward pass, with a matched-rank random-
#              direction control and a zero-ablation upper bound. Mutated-slot
#              queries carry a real veracity alternative (flip readout);
#              untouched-slot queries do not and act as the built-in
#              retrieval-specificity control.
#              Gemma needs --quote_prefill: its tokenizer has no merged 'X
#              tokens, so without prefill both answers share the bare '
#              first token and the flip readout is degenerate. The subspace
#              is refit in-job from this run's own activations, so the
#              prefill change cannot desynchronise it from the intervention.
#
# Tasks 0/1 are the prerequisite for the position-prediction control but NOT
# for 2/3 — all four can run concurrently.

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

STIM=results/rsa_head_outputs/stimuli_code_padtwins
if [ ! -f $STIM/code_padtwins.csv ]; then
  python analysis_scripts/rsa_head_outputs/build_code_padtwins_stimuli.py || exit 1
fi

ATTN_OUT=results/rsa_head_outputs/activations_code_padtwins_attn

case "$SLURM_ARRAY_TASK_ID" in
  0)
    echo "Task 0: attention-summary capture, Qwen2.5-14B-Instruct"
    python analysis_scripts/rsa_head_outputs/capture_head_outputs_full.py \
      --template code_padtwins \
      --stim_dir  $STIM \
      --out_dir   $ATTN_OUT/Qwen2.5-14B-Instruct \
      --model_type Qwen2.5-14B-Instruct \
      --system_prompt code \
      --capture_attn_summary
    ;;
  1)
    echo "Task 1: attention-summary capture, gemma-2-27b-it"
    python analysis_scripts/rsa_head_outputs/capture_head_outputs_full.py \
      --template code_padtwins \
      --stim_dir  $STIM \
      --out_dir   $ATTN_OUT/gemma-2-27b-it \
      --model_type gemma-2-27b-it \
      --ungroup_grouped_query_attention \
      --system_prompt code \
      --capture_attn_summary
    ;;
  2)
    echo "Task 2: causal pad-subspace ablation, Qwen2.5-14B-Instruct"
    python analysis_scripts/rsa_head_outputs/padtwins_causal_ablation.py \
      --model qwen --k 2 --random_heads
    ;;
  3)
    echo "Task 3: causal pad-subspace ablation, gemma-2-27b-it"
    python analysis_scripts/rsa_head_outputs/padtwins_causal_ablation.py \
      --model gemma --k 2 --random_heads --quote_prefill
    ;;
esac
