#!/bin/bash
# GPT-OSS DAPO Training Script with DeepSpeed

echo "🚀 Starting GPT-OSS DAPO Training on GPU 6"
echo "=========================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="39550"
export WORLD_SIZE="1"
export RANK="0"

# Activate conda environment with dependencies
source /data/miniforge3/etc/profile.d/conda.sh
conda activate gpt_oss_rl

# Navigate to project directory
cd /home/ubuntu/gpt_oss_arc_final

# Run training
python main_hf_trl_dapo.py 2>&1 | tee dapo_training_$(date +%Y%m%d_%H%M%S).log

echo "Training completed!"
