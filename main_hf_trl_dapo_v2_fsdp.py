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
from trl import GRPOTrainer, GRPOConfig
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
    """Create training prompt using tokenizer.apply_chat_template with reasoning_effort=medium.

    Harmony/chat-template only. Do not manually inject channel tokens.
    """
    
    # Convert grids to strings
    train_examples_str = []
    for i, train_pair in enumerate(problem.train_pairs):
        input_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in train_pair.x)
        output_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in train_pair.y)
        train_examples_str.append(f"Example {i+1}:\nInput:\n{input_str}\n\nOutput:\n{output_str}")
    
    # Test input
    test_input = problem.test_pairs[0].x
    test_input_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in test_input)
    
    # Expected output for training
    test_output = problem.test_pairs[0].y
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
        {"role": "system", "content": "You are a helpful assistant specializing in solving ARC puzzles. Provide concise reasoning and a final grid answer."},
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

    # No manual channel tokens; rely on Harmony/chat template for stops
    expected_output = test_output_str
    return prompt, expected_output

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
        for i in range(min(num_problems, len(train_problems))):
            problem = train_problems[i]
            problem_id = f"train_{i:03d}"
                
            try:
                prompt, completion = create_training_prompt(problem, tokenizer)
                
                self.problems.append({
                    'problem_id': problem_id,
                    'prompt': prompt,
                    'expected_output': completion  # store for optional rewards/analysis
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
    ignored = []
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            ignored.append(emb)
    except Exception:
        pass
    try:
        out_emb = model.get_output_embeddings()
        if out_emb is not None:
            ignored.append(out_emb)
    except Exception:
        pass

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,  # Required for LoRA
        ignored_modules=ignored if ignored else None,
    )
    
    log_with_timestamp(f"✅ Model wrapped with FSDP")
    return model

def load_model_and_tokenizer(checkpoint_path=None):
    """Load GPT-OSS model and tokenizer with FSDP support and optional checkpoint loading."""
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
    if checkpoint_path and os.path.exists(checkpoint_path):
        log_with_timestamp(f"🔄 Loading model from checkpoint: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None,  # Let FSDP handle device placement
        )
        log_with_timestamp(f"✅ Model loaded from checkpoint: {model.num_parameters():,} parameters")
    else:
        log_with_timestamp(f"🔄 Loading base model from HuggingFace: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None,  # Let FSDP handle device placement
        )
        log_with_timestamp(f"✅ Base model loaded: {model.num_parameters():,} parameters")
        
        # Apply LoRA only for base model (checkpoint already has LoRA)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model = get_peft_model(model, lora_config)
        try:
            trainable_params, total_params = model.get_nb_trainable_parameters()
            pct = (trainable_params / total_params * 100) if total_params else 0.0
            log_with_timestamp(
                f"✅ LoRA applied: {total_params:,} total, {trainable_params:,} trainable ({pct:.4f}%)"
            )
        except Exception:
            # Fallback if API differs
            log_with_timestamp("✅ LoRA applied")

        # Keep weights in default dtype; Accelerate/Trainer handles mixed precision (bf16)
    
    # Let HF/Accelerate handle FSDP wrapping via trainer args to avoid PEFT/FSDP param shape issues
    
    return model, tokenizer

def run_fsdp_training(checkpoint_path=None, output_dir=None):
    """Run GRPO training with FSDP and checkpoint support."""
    
    # Check if distributed is initialized
    if not dist.is_initialized():
        log_with_timestamp("❌ Distributed training not initialized. Running single GPU mode.")
        rank = 0
    else:
        rank = dist.get_rank()
    
    if rank == 0:
        log_with_timestamp("🚀 STARTING FSDP GRPO TRAINING")
        log_with_timestamp("🖥️ Multi-GPU FSDP training with GRPO")
        if checkpoint_path:
            log_with_timestamp(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        if output_dir:
            log_with_timestamp(f"💾 Output directory: {output_dir}")
    
    # Setup W&B
    config = setup_wandb()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)
    
    # Create dataset
    dataset = ARCDatasetFSDP(tokenizer, max_length=4096, num_problems=10)
    
    if len(dataset.problems) == 0:
        log_with_timestamp("❌ No training problems loaded!")
        return
    
    # Training arguments with FSDP settings (for logging config only)
    training_args = TrainingArguments(
        output_dir="checkpoints_fsdp_fixed",
        overwrite_output_dir=False,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=50,
        learning_rate=5e-6,
        bf16=False,
        logging_steps=1,
        logging_dir="checkpoints_fsdp_fixed/logs",
        save_steps=25,
        save_total_limit=None,
        gradient_checkpointing=False,
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
    
    def compute_reward(completions, prompts, **kwargs):
        """Heuristic reward: prefer grid-like outputs (digits/spaces, multiple lines)."""
        import re
        rewards = []
        grid_line = re.compile(r"^[0-9 ]+$")
        for completion in completions:
            # Basic heuristic: multiple lines and lines look like grids of digits
            lines = [ln.strip() for ln in completion.strip().splitlines() if ln.strip()]
            if len(lines) >= 1 and all(grid_line.match(ln) for ln in lines):
                reward = 1.0
            elif len(completion.strip()) > 0:
                reward = 0.3
            else:
                reward = 0.0
            rewards.append(reward)
        return rewards
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = "checkpoints_fsdp_fixed"
    
    # Create output directory if it doesn't exist
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        log_with_timestamp(f"📁 Output directory created: {output_dir}")
    
    # GRPO configuration 
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=50,
        warmup_steps=5,
        logging_steps=1,
        save_steps=25,  # Save checkpoint every 25 steps
        save_total_limit=3,  # Keep only last 3 checkpoints
        bf16=False,
        gradient_checkpointing=False,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "min_num_params": 1000000,
            "activation_checkpointing": True,
            "xla": False,
        },
        # Resume from checkpoint
        resume_from_checkpoint=checkpoint_path if checkpoint_path and os.path.exists(checkpoint_path) else None,
        # GRPO specific
        num_generations=2,
        generation_batch_size=2,
        max_prompt_length=4096,
        max_completion_length=12000,  # Reduced from 30k for stability
        num_iterations=1,
        epsilon=0.2,
        beta=0.0,
        loss_type="bnpo",
        temperature=0.7,
        top_p=0.9,
        report_to=["wandb"] if rank == 0 else [],
    )
    
    log_with_timestamp("🚀 Initializing GRPO trainer with FSDP...")
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[compute_reward],  # Required for GRPO
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer
    )
    
    if rank == 0:
        log_with_timestamp(f"✅ GRPO trainer initialized")
        log_with_timestamp(f"📊 Model parameters: {model.num_parameters():,}")
        log_with_timestamp(f"📊 Training dataset size: {len(dataset)}")
        log_with_timestamp(f"🎯 Max steps: {grpo_config.max_steps}")
        log_with_timestamp("================================================================================")
    
    # Start training
    log_with_timestamp("🚀 Starting training...")
    
    try:
        trainer.train()
        log_with_timestamp("✅ Training completed successfully!")
        
        if rank == 0:
            # Save final model
            final_checkpoint_path = os.path.join(output_dir, "final_checkpoint")
            trainer.save_model(final_checkpoint_path)
            log_with_timestamp(f"💾 Final model saved to: {final_checkpoint_path}")
            
            # Also save tokenizer
            tokenizer.save_pretrained(final_checkpoint_path)
            log_with_timestamp(f"💾 Tokenizer saved to: {final_checkpoint_path}")
            
            # Save training info
            training_info = {
                "model_name": "openai/gpt-oss-20b",
                "training_type": "GRPO_FSDP",
                "max_steps": grpo_config.max_steps,
                "learning_rate": grpo_config.learning_rate,
                "lora_r": 16,
                "lora_alpha": 32,
                "completion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_problems": len(dataset.problems),
                "world_size": dist.get_world_size() if dist.is_initialized() else 1
            }
            
            with open(os.path.join(final_checkpoint_path, "training_info.json"), "w") as f:
                json.dump(training_info, f, indent=2)
            log_with_timestamp(f"💾 Training info saved to: {final_checkpoint_path}/training_info.json")
            
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_with_timestamp(f"❌ Training failed: {e}")
        # Persist full traceback for debugging
        try:
            rank_str = f"rank{rank}" if 'rank' in locals() else "rankx"
            debug_path = os.path.join(output_dir if output_dir else '.', f"error_{rank_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            with open(debug_path, 'w') as f:
                f.write(tb)
            log_with_timestamp(f"📝 Full traceback saved: {debug_path}")
        except Exception:
            pass
        raise
    finally:
        if rank == 0 and wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-OSS GRPO Training with FSDP")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum training steps")
    parser.add_argument("--save_steps", type=int, default=25, help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    
    # Setup distributed training
    local_rank = setup_distributed()
    
    # Set seed for reproducibility
    set_seed(42)
    
    try:
        run_fsdp_training(
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        log_with_timestamp("⚠️ Training interrupted by user")
    except Exception as e:
        log_with_timestamp(f"❌ Training failed: {e}")
        sys.exit(1)
