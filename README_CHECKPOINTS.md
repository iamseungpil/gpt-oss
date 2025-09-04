# GPT-OSS Checkpoint Management Guide

## 🚀 Training with Checkpoints

### Basic Training (New Model)
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=6 python main_hf_trl_dapo_v2.py

# Multi-GPU FSDP
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=29600 main_hf_trl_dapo_v2_fsdp.py \
    --output_dir /path/to/checkpoints \
    --max_steps 100 \
    --save_steps 25
```

### Resume from Checkpoint
```bash
# Resume FSDP training from checkpoint
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=29600 main_hf_trl_dapo_v2_fsdp.py \
    --checkpoint_path /path/to/checkpoints/checkpoint-50 \
    --output_dir /path/to/checkpoints \
    --max_steps 100
```

## 💾 Checkpoint Structure

When training completes, checkpoints are saved with this structure:
```
/path/to/checkpoints/
├── checkpoint-25/           # Intermediate checkpoint at step 25
│   ├── pytorch_model.bin
│   ├── training_args.bin
│   └── trainer_state.json
├── checkpoint-50/           # Intermediate checkpoint at step 50
└── final_checkpoint/        # Final model after training completion
    ├── pytorch_model.bin    # Trained model weights
    ├── tokenizer.json       # Tokenizer files
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── config.json          # Model configuration
    └── training_info.json   # Training metadata
```

## 🔄 Loading Checkpoints

### For Inference
```python
from load_checkpoint_example import load_checkpoint_for_inference

# Load the final trained model
model, tokenizer = load_checkpoint_for_inference("/path/to/checkpoints/final_checkpoint")

# Test with ARC puzzle
response = test_inference(model, tokenizer, "Solve this ARC pattern recognition task.")
```

### Command Line Testing
```bash
# Test a checkpoint with custom prompt
python load_checkpoint_example.py /path/to/checkpoints/final_checkpoint \
    --test_prompt "Find the pattern in this ARC puzzle and solve it."
```

### For Continued Training
```bash
# Resume training from intermediate checkpoint
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=29600 main_hf_trl_dapo_v2_fsdp.py \
    --checkpoint_path /path/to/checkpoints/checkpoint-50 \
    --output_dir /path/to/new_checkpoints \
    --max_steps 200 \
    --save_steps 25
```

## 📋 Training Info

Each final checkpoint includes `training_info.json` with:
- Model name and type
- Training parameters (steps, learning rate, LoRA config)
- Completion timestamp  
- Dataset information
- Distributed training details

## 🔧 Configuration Options

### Training Arguments
- `--checkpoint_path`: Path to resume from (optional)
- `--output_dir`: Directory to save checkpoints (default: auto-generated)
- `--max_steps`: Total training steps (default: 50)
- `--save_steps`: Save frequency (default: 25)

### Automatic Features
- **Auto-save**: Checkpoints saved every `save_steps`
- **Resume**: Automatically detect and resume from checkpoint
- **Cleanup**: Keep only last 3 checkpoints (`save_total_limit=3`)
- **Metadata**: Training info saved with each final checkpoint

## 🎯 Channel Switching Training

The GRPO training is optimized for ARC puzzle solving with:
- **Reward Function**: Higher rewards for proper `<|channel|>analysis<|message|>` and `<|channel|>final<|message|>` usage
- **30k Token Support**: Training with extended context for complex reasoning
- **LoRA Fine-tuning**: Efficient parameter updates (0.04% of total parameters)
- **FSDP Support**: Distributed training across multiple GPUs

## ⚡ Quick Start Examples

```bash
# Start fresh training
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=29600 main_hf_trl_dapo_v2_fsdp.py

# Resume from last checkpoint  
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=29600 main_hf_trl_dapo_v2_fsdp.py \
    --checkpoint_path ./checkpoints_hf_trl_grpo_v2_fsdp/checkpoint-50

# Test trained model
python load_checkpoint_example.py ./checkpoints_hf_trl_grpo_v2_fsdp/final_checkpoint
```