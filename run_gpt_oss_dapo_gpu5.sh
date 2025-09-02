#!/bin/bash
# GPT-OSS DAPO Training Script on GPU 5 (gptoss env, Harmony prompting, no forced tokens)

set -euo pipefail

echo "🚀 Starting GPT-OSS DAPO Training on GPU 5 (gptoss)"
echo "===================================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="39555"
export WORLD_SIZE="1"
export RANK="0"

# Activate conda environment
source /data/miniforge3/etc/profile.d/conda.sh
conda activate gptoss

# Navigate to project directory
cd "$(dirname "$0")"

# Run training (Harmony-based prompting, no forced final injection)
ts=$(date +%Y%m%d_%H%M%S)
python -u main_hf_trl_dapo.py 2>&1 | tee "dapo_training_gpu5_harmony_${ts}.log"

echo "✅ Training command launched. Logs: dapo_training_gpu5_harmony_${ts}.log"

