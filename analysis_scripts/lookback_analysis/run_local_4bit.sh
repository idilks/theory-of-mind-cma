#!/bin/bash
# Run the lookback capture LOCALLY with 4-bit Qwen2.5-14B-Instruct.
#
# Memory: ~9GB for weights + ~3GB activations. Fits on:
#   - Apple Silicon w/ ≥16GB unified memory
#   - Any 12GB+ CUDA GPU
#
# Requires: bitsandbytes installed in your active venv.
#   pip install bitsandbytes accelerate
#
# Usage:
#   bash analysis_scripts/lookback_analysis/run_local_4bit.sh
# or one template at a time:
#   bash analysis_scripts/lookback_analysis/run_local_4bit.sh food_truck

set -euo pipefail

cd "$(dirname "$0")/../.."

# Pick device_map: cuda if available, else auto (will use mps on Apple Silicon)
DEVICE_MAP="${DEVICE_MAP:-auto}"

TEMPLATES=("${1:-food_truck}" hair_styling library_book)
if [ -n "${1:-}" ]; then
  TEMPLATES=("$1")
fi

for T in "${TEMPLATES[@]}"; do
  echo "=== capturing template=$T ==="
  python analysis_scripts/lookback_analysis/capture_attention_patterns.py \
    --template "$T" \
    --condition abstract_belief \
    --base_rule ABA \
    --exp_rule ABB \
    --patch_position patch_before_movement \
    --top_k 50 \
    --prompt_num 50 \
    --seed 0 \
    --results_root results \
    --out_root local_results/lookback_analysis \
    --device_map "$DEVICE_MAP" \
    --load_in_4bit

  CAP_DIR="local_results/lookback_analysis/abstract_belief/template_${T}/base_rule_ABA_exp_rule_ABB/patch_before_movement"
  echo "=== analyzing $CAP_DIR ==="
  python analysis_scripts/lookback_analysis/analyze_lookback_attention.py --capture_dir "$CAP_DIR"
done
