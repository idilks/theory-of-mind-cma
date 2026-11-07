#!/bin/bash
#SBATCH --job-name=region_attn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --partition=standard
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/region_attn_%j.out
#SBATCH --error=logs/region_attn_%j.err

# Per-head share of final-token attention in each prompt region (original location,
# current location, question, rest of story, scaffold), averaged over ALL prompts,
# for every captured head in every cell of the belief and photo conditions:
#   belief: C1 (divergence) + C5 (retrieval) heads on C1 prompts, false- and true-belief
#   photo:  C2 (divergence) + C6 (retrieval) heads on C2 prompts, outdated and current photo
# CPU-only; reads the matched captures from lookback_capture_matched.sh.
# Output (small CSV, pulled by pull_results.sh block 2):
#   results/lookback_analysis/_regions/region_shares_matched.csv
# The code domain (K1 vs K5) is computed locally from
# results/rsa_head_outputs/activations_code_padtwins_attn/, so it is not run here.

export LD_LIBRARY_PATH=/dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis/lib:$LD_LIBRARY_PATH
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HUB_OFFLINE=1

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis
mkdir -p logs

# Reads the matched captures from lookback_capture_matched.sh (both head groups on the
# same prompts, false- and true-belief); run that job first.
python analysis_scripts/lookback_analysis/region_attention_shares.py \
  --root results/lookback_analysis/_matched \
  --conditions abstract_belief abstract_photo \
  --vocab_dir tom_datasets \
  --out results/lookback_analysis/_regions/region_shares_matched.csv
