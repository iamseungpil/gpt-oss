#!/usr/bin/env python3
"""
GPT-OSS 20B ARC Training with DAPO (Fixed Version)
===================================================

Fixing compatibility issues between GPT-OSS and Unsloth's GRPO implementation.
Based on GitHub issue #1624 and documentation.
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import re

# Environment setup - CRITICAL for GPT-OSS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["WANDB_API_KEY"] = "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.disable = True

# Configuration
DATA_DIR = Path("/data/gpt_oss_final")
LOG_FILE = DATA_DIR / "logs" / "dapo_fixed.log"
RESPONSE_DIR = DATA_DIR / "logs" / "dapo_responses_fixed"

# Setup logging
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_grid_from_response(response: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Extract grid from model response"""
    
    # Try to find grid in code blocks
    code_block_pattern = r'```(?:python|text|)?\n?([\s\S]*?)```'
    code_blocks = re.findall(code_block_pattern, response, re.MULTILINE)
    
    for block in code_blocks:
        lines = block.strip().split('\n')
        grid = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse numbers from line
            numbers = []
            for item in line.replace(',', ' ').split():
                try:
                    num = int(item)
                    numbers.append(num)
                except ValueError:
                    continue
            
            if numbers:
                grid.append(numbers)
        
        # Check if valid grid
        if grid and len(grid) == target_shape[0]:
            if all(len(row) == target_shape[1] for row in grid):
                return np.array(grid)
    
    return None

def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format"""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

def compute_arc_reward(response: str, target_grid: np.ndarray) -> float:
    """Compute reward for ARC task"""
    
    parsed_grid = parse_grid_from_response(response, target_grid.shape)
    
    if parsed_grid is None:
        return -1.0
    
    # Perfect match
    if np.array_equal(parsed_grid, target_grid):
        return 1.0
    
    # Partial credit for correct cells
    correct_cells = np.sum(parsed_grid == target_grid)
    total_cells = target_grid.size
    accuracy = correct_cells / total_cells
    
    # Scale reward
    if accuracy > 0.9:
        return 0.5
    elif accuracy > 0.7:
        return 0.2
    else:
        return -0.5

def create_arc_prompt(problem, use_channel_markers: bool = True) -> str:
    """Create ARC prompt with Harmony format"""
    
    # Build examples section
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(f"Example {i}: Input:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}\n")
    
    # Test section
    test_input = problem.test_pairs[0].x
    examples_text = '\n'.join(examples)
    
    if use_channel_markers:
        # With proper Harmony format
        prompt = f"""<|start|>system<|message|>You are a pattern solver. Use high reasoning.
Reasoning: high
Output format: Provide the final answer as a grid of numbers.<|end|>
<|start|>user<|message|>Solve this pattern and output the grid:
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|message|>"""
    else:
        # Simplified format
        prompt = f"""You are a pattern solver. Solve this pattern:
{examples_text}
Test Input:
{grid_to_string(test_input)}

Output the solution grid:"""
    
    return prompt

def main_dapo_fixed():
    """Main DAPO training with fixes"""
    
    logger.info("🚀 Starting GPT-OSS ARC DAPO Training (Fixed)")
    logger.info("="*50)
    
    try:
        # CRITICAL: Import Unsloth components
        from unsloth import FastLanguageModel, PatchFastRL
        from unsloth import is_bfloat16_supported
        
        # CRITICAL: Patch before importing GRPOTrainer
        PatchFastRL("GRPO", FastLanguageModel)
        
        from trl import GRPOConfig, GRPOTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
        from arc import train_problems
        import wandb
        
        # Setup WandB
        wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
        wandb.init(
            project="gpt-oss-arc-dapo-fixed",
            name=f"dapo_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        
        # Load model with CRITICAL settings
        logger.info("Loading model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gpt-oss-20b-BF16",
            max_seq_length=2048,
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float32,
            load_in_4bit=True,
            device_map={"": 0},  # Single GPU mapping
            trust_remote_code=True,
            # fast_inference requires vLLM, using default
        )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=False,  # Disabled for compatibility
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Setup generation config
        model.generation_config.max_new_tokens = 512  # Reduced for stability
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.7
        model.generation_config.top_p = 0.9
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✅ Model loaded | Trainable params: {trainable_params/1e6:.2f}M")
        
        # Prepare dataset
        problems = train_problems[:5]  # Start with fewer problems
        logger.info(f"📊 Loaded {len(problems)} ARC problems")
        
        # Create prompts
        prompts = []
        targets = []
        
        for problem in problems:
            prompt = create_arc_prompt(problem, use_channel_markers=True)
            prompts.append(prompt)
            targets.append(problem.test_pairs[0].y)
        
        # Create dataset
        dataset_dict = {
            "prompt": prompts,
            "problem_id": [p.uid for p in problems]
        }
        
        train_dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"📊 Created dataset with {len(train_dataset)} examples")
        
        # GRPO Config - Unsloth specific configuration
        grpo_config = GRPOConfig(
            use_vllm=False,  # Disable vLLM for now - GPT-OSS compatibility issue
            learning_rate=1e-5,
            num_generations=2,  # Very conservative start
            max_steps=50,  # Limit steps for testing
            epsilon=0.2,
            epsilon_high=0.28,
            delta=1.5,
            beta=0.0,  # DAPO paper recommends beta=0 to remove KL term
            loss_type='dapo',  # Use DAPO loss
            mask_truncated_completions=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            logging_steps=1,
            save_steps=10,
            fp16=False,
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            report_to="wandb",
            torch_compile=False,
            output_dir=str(DATA_DIR / "checkpoints_dapo_fixed"),
        )
        
        # Custom reward function - updated to accept all parameters
        def reward_function(completions: List[str], prompts: List[str], completion_ids: List = None, **kwargs) -> List[float]:
            """Compute rewards for completions"""
            rewards = []
            
            for i, (completion, prompt) in enumerate(zip(completions, prompts)):
                # Get corresponding target
                target_idx = i % len(targets)
                target_grid = targets[target_idx]
                
                # Compute reward
                reward = compute_arc_reward(completion, target_grid)
                rewards.append(reward)
                
                logger.info(f"Reward for completion {i}: {reward:.2f}")
            
            return rewards
        
        # Create trainer - Unsloth's GRPOTrainer has specific interface
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_function,
            args=grpo_config,  # args, not config
            train_dataset=train_dataset,
            processing_class=tokenizer,  # tokenizer as processing_class
        )
        
        logger.info("🚀 Starting DAPO training with fixes...")
        
        # Train - without callback for now
        trainer.train()
        
        logger.info("✅ Training completed!")
        
        # Save model
        model.save_pretrained(DATA_DIR / "final_model_dapo_fixed")
        tokenizer.save_pretrained(DATA_DIR / "final_model_dapo_fixed")
        logger.info(f"💾 Model saved to {DATA_DIR / 'final_model_dapo_fixed'}")
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        wandb.finish()

if __name__ == "__main__":
    main_dapo_fixed()