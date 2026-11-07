#!/bin/bash
#SBATCH --job-name=rsa_content_analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=standard,preempt_wager,preempt_lsong
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/rsa_content_analyze_%j.out
#SBATCH --error=logs/rsa_content_analyze_%j.err

# CPU-only RSA + MVPA on the content-separable ToM capture, run ON the cluster
# so the three ~180 MB .pt files never need to come down. Chain it after the
# capture array:
#
#   CAP=$(sbatch --parsable slurm_scripts/rsa_head_capture_content.sh)
#   sbatch --dependency=afterok:$CAP slurm_scripts/rsa_content_analyze.sh
#
# Writes only small artifacts, which pull_results.sh brings back:
#   results/rsa_head_outputs/aggregation_content/*.csv        (block 4)
#   results/rsa_head_outputs/mvpa_veracity/*.csv              (block 4)
#   results/summary/aggregation/content_rdm_bars.png           (block 6)
# The .pt captures stay on the cluster (pull_results.sh excludes
# activations_content/*.pt).

export LD_LIBRARY_PATH=/dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis/lib:$LD_LIBRARY_PATH
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

ACT_DIR=results/rsa_head_outputs/activations_content
for t in food_truck hair_styling library_book; do
  [ -f $ACT_DIR/$t.pt ] || { echo "missing $ACT_DIR/$t.pt"; exit 1; }
done

python -c "import sklearn" 2>/dev/null || pip install --quiet scikit-learn

echo "=== content RDMs ==="
python analysis_scripts/rsa_head_outputs/analyze_content_rdms.py --n_perm 5000 --n_random 100

echo "=== MVPA: veracity ==="
python analysis_scripts/rsa_head_outputs/mvpa_veracity.py --act_dir $ACT_DIR \
  --out_dir results/rsa_head_outputs/mvpa_veracity/content_capture --target belief_veracity

echo "=== MVPA: believed location (6-way) ==="
python analysis_scripts/rsa_head_outputs/mvpa_veracity.py --act_dir $ACT_DIR \
  --out_dir results/rsa_head_outputs/mvpa_veracity/content_capture --target believed

echo "done"
