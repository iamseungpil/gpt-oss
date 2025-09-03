#!/usr/bin/env python3
"""
GPT-OSS Training with HuggingFace TRL GRPO/DAPO v2 - FSDP version for multi-GPU training
"""

import os
import sys
import json
import time
import wandb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# FSDP and distributed imports
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader

# Core ML imports
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed
)

# TRL imports for GRPO
from trl import DAPOTrainer, DAPOConfig, setup_chat_format
from peft import LoraConfig, get_peft_model, TaskType

# ARC dataset
try:
    from arc import train_problems
    HAS_ARC = True
except ImportError:
    HAS_ARC = False
    print("❌ ARC dataset not available")

def log_with_timestamp(message):
    """Enhanced logging with timestamp and rank."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[{timestamp}] [Rank {rank}] {message}")
    sys.stdout.flush()

def setup_distributed():
    """Setup distributed training environment with NCCL optimizations."""
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "2"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29600"
    
    # Set NCCL environment variables for better stability
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes timeout
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand
    os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P for stability
    os.environ['NCCL_SOCKET_NTHREADS'] = '1'
    os.environ['NCCL_NSOCKS_PERTHREAD'] = '1'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    try:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=60)  # Extended timeout
        )
        
        # Test basic communication
        if dist.is_initialized():
            test_tensor = torch.tensor([local_rank], device=f'cuda:{local_rank}')
            dist.all_reduce(test_tensor)
            log_with_timestamp(f"✅ NCCL communication test passed, sum: {test_tensor.item()}")
        
        log_with_timestamp(f"🚀 FSDP Distributed setup complete - Rank: {dist.get_rank()}/{dist.get_world_size()}")
        return local_rank
    except Exception as e:
        log_with_timestamp(f"❌ NCCL initialization failed: {e}")
        # Fallback to single GPU mode
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        log_with_timestamp("⚠️ Falling back to single GPU mode")
        return local_rank

def setup_wandb():
    """Initialize Weights & Biases with project config."""
    config = {
        'model': 'openai/gpt-oss-20b',
        'method': 'HF TRL GRPO/DAPO',
        'max_tokens': 30000,
        'memory_optimized': True,
        'reward_components': 5,
        'dataset_size': 10,
        'continual_learning': True,
        'steps_per_problem': 50,
        'fsdp_enabled': True,
        'world_size': dist.get_world_size() if dist.is_initialized() else 1
    }
    
    # Only initialize wandb on rank 0
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.init(
            project="gpt-oss-arc-training-fsdp",
            name=f"dapo-v2-fsdp-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config,
            tags=["GPT-OSS", "ARC", "DAPO", "FSDP", "20B", "reasoning", "puzzle-solving"]
        )
        log_with_timestamp("✅ W&B initialized")
    
    return config

def create_training_prompt(problem, tokenizer):
    """Create training prompt using tokenizer.apply_chat_template with reasoning_effort=medium."""
    
    # Convert grids to strings
    train_examples_str = []
    for i, (input_grid, output_grid) in enumerate(problem['train']):
        input_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in input_grid)
        output_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in output_grid)
        train_examples_str.append(f"Example {i+1}:\nInput:\n{input_str}\n\nOutput:\n{output_str}")
    
    # Test input
    test_input = problem['test'][0]['input']
    test_input_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in test_input)
    
    # Expected output for training
    test_output = problem['test'][0]['output'] 
    test_output_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in test_output)
    
    user_content = f"""# ARC Puzzle Solver

You are an expert at solving Abstract Reasoning Corpus (ARC) puzzles. Given training examples, find the pattern and apply it to the test input.

## Training Examples:
{chr(10).join(train_examples_str)}

## Test Input:
{test_input_str}

Please analyze the pattern and provide the solution."""
    
    # Create messages for chat template
    messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. You are chatting with the user via the ChatGPT iOS app. This means most of the time your lines should be a sentence or two, unless the user's request requires reasoning or long-form outputs. Never use emojis, unless explicitly asked to. Knowledge cutoff: 2023-10 Current date: 2024-07-07 Reasoning: medium"},
        {"role": "user", "content": user_content}
    ]
    
    # Use tokenizer.apply_chat_template with reasoning_effort="medium"
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="medium"
        )
    except Exception as e:
        log_with_timestamp(f"⚠️ reasoning_effort not supported: {e}")
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    # Add channel start for GPT-OSS assistant response
    prompt = prompt + "<|channel|>"
    
    # Create completion with channel switching and final answer
    completion = f"<|channel|>analysis<|message|>\n\nLet me analyze the pattern in these examples...\n\n<|channel|>final<|message|>\n\n{test_output_str}<|return|>"
    
    return prompt, completion

class ARCDatasetFSDP(Dataset):
    """ARC dataset for DAPO training with FSDP support."""
    
    def __init__(self, tokenizer, max_length=4096, num_problems=10):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.problems = []
        
        if not HAS_ARC:
            log_with_timestamp("❌ ARC dataset not available")
            return
            
        # Use first num_problems from train_problems
        for i, (problem_id, problem_data) in enumerate(train_problems.items()):
            if i >= num_problems:
                break
                
            try:
                prompt, completion = create_training_prompt(problem_data, tokenizer)
                
                self.problems.append({
                    'problem_id': problem_id,
                    'prompt': prompt,
                    'completion': completion
                })
                
            except Exception as e:
                log_with_timestamp(f"⚠️ Error processing problem {problem_id}: {e}")
                continue
        
        log_with_timestamp(f"✅ Loaded {len(self.problems)} training problems")
    
    def __len__(self):
        return len(self.problems) * 50  # 50 steps per problem
    
    def __getitem__(self, idx):
        # Cycle through problems
        problem_idx = idx % len(self.problems)
        problem = self.problems[problem_idx]
        
        return {
            'prompt': problem['prompt'],
            'completion': problem['completion'],
            'problem_id': problem['problem_id']
        }

def wrap_model_with_fsdp(model):
    """Wrap model with FSDP for distributed training."""
    from functools import partial
    
    # Auto-wrap policy for FSDP
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=1000000  # 1M parameters minimum per shard
    )
    
    # Mixed precision configuration
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,  # Required for LoRA
    )
    
    log_with_timestamp(f"✅ Model wrapped with FSDP")
    return model

def load_model_and_tokenizer():
    """Load GPT-OSS model and tokenizer with FSDP support."""
    log_with_timestamp("📦 Loading GPT-OSS model and tokenizer...")
    
    model_name = "openai/gpt-oss-20b"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='left'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    log_with_timestamp("✅ Tokenizer loaded")
    
    # Load model with bfloat16 precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,  # Let FSDP handle device placement
    )
    
    log_with_timestamp(f"✅ Base model loaded: {model.num_parameters():,} parameters")
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    log_with_timestamp(f"✅ LoRA applied: {model.num_parameters():,} total, {model.get_nb_trainable_parameters():,} trainable ({model.get_nb_trainable_parameters()/model.num_parameters()*100:.4f}%)")
    
    # Wrap with FSDP
    if dist.is_initialized():
        model = wrap_model_with_fsdp(model)
    
    return model, tokenizer

def run_fsdp_training():
    """Run DAPO training with FSDP."""
    
    # Check if distributed is initialized
    if not dist.is_initialized():
        log_with_timestamp("❌ Distributed training not initialized. Running single GPU mode.")
        rank = 0
    else:
        rank = dist.get_rank()
    
    if rank == 0:
        log_with_timestamp("🚀 STARTING FSDP DAPO TRAINING")
        log_with_timestamp("🖥️ Multi-GPU FSDP training with GRPO/DAPO")
    
    # Setup W&B
    config = setup_wandb()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Create dataset
    dataset = ARCDatasetFSDP(tokenizer, max_length=4096, num_problems=10)
    
    if len(dataset.problems) == 0:
        log_with_timestamp("❌ No training problems loaded!")
        return
    
    # Training arguments with FSDP settings
    training_args = TrainingArguments(
        output_dir="/opt/dlami/nvme/gpt_oss/checkpoints_hf_trl_dapo_v2_fsdp",
        overwrite_output_dir=False,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=50,
        learning_rate=5e-6,
        bf16=True,
        logging_steps=1,
        logging_dir="/opt/dlami/nvme/gpt_oss/checkpoints_hf_trl_dapo_v2_fsdp/logs",
        save_steps=25,
        save_total_limit=None,
        gradient_checkpointing=True,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        report_to=["wandb"] if rank == 0 else [],
        disable_tqdm=False,
        skip_memory_metrics=True,
        # FSDP settings
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "min_num_params": 1000000,
            "xla": False,
            "xla_fsdp_v2": False,
            "xla_fsdp_grad_ckpt": False
        },
    )
    
    # DAPO configuration
    dapo_trainer_config = {
        'beta': 0.0,
        'num_generations': 2,
        'generation_batch_size': 2,
        'steps_per_generation': 2,
        'max_completion_length': 30000,
        'max_prompt_length': 4096,
        'temperature': 0.7,
        'top_p': 0.9,
        'loss_type': 'bnpo',
        'epsilon': 0.2,
        'epsilon_high': 0.28,
        'ref_model_mixup_alpha': 0.6,
        'ref_model_sync_steps': 512,
        'scale_rewards': True,
        'mask_truncated_completions': False,
        'sync_ref_model': False,
        'use_liger_loss': False,
        'log_completions': False,
        'shuffle_dataset': True,
        'ds3_gather_for_generation': True,
    }
    
    log_with_timestamp("🚀 Initializing DAPO trainer with FSDP...")
    
    # Initialize trainer
    trainer = DAPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        **dapo_trainer_config
    )
    
    if rank == 0:
        log_with_timestamp(f"✅ DAPO trainer initialized")
        log_with_timestamp(f"📊 Model parameters: {model.num_parameters():,}")
        log_with_timestamp(f"📊 Training dataset size: {len(dataset)}")
        log_with_timestamp(f"🎯 Max steps: {training_args.max_steps}")
        log_with_timestamp("================================================================================")
    
    # Start training
    log_with_timestamp("🚀 Starting training...")
    
    try:
        trainer.train()
        log_with_timestamp("✅ Training completed successfully!")
        
        if rank == 0:
            # Save final model
            trainer.save_model()
            log_with_timestamp("💾 Final model saved")
            
    except Exception as e:
        log_with_timestamp(f"❌ Training failed: {e}")
        raise
    finally:
        if rank == 0 and wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-OSS DAPO Training with FSDP")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Setup distributed training
    local_rank = setup_distributed()
    
    # Set seed for reproducibility
    set_seed(42)
    
    try:
        run_fsdp_training()
    except KeyboardInterrupt:
        log_with_timestamp("⚠️ Training interrupted by user")
    except Exception as e:
        log_with_timestamp(f"❌ Training failed: {e}")
        sys.exit(1)