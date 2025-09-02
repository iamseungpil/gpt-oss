#!/usr/bin/env python3
"""
GPT-OSS 20B ARC Training with Standard TRL GRPO
==============================================

Using standard TRL GRPOTrainer without Unsloth to avoid compatibility issues.
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

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["WANDB_API_KEY"] = "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"
# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Core imports
import wandb
from arc import train_problems
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
DATA_DIR = Path("/data/gpt_oss_final")
LOG_FILE = DATA_DIR / "logs" / "grpo_standard.log"
RESPONSE_DIR = DATA_DIR / "logs" / "grpo_responses"

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

def create_arc_prompt(problem) -> str:
    """Create ARC prompt with Harmony format"""
    
    # Build examples section
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(f"Example {i}: Input:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}\n")
    
    # Test section
    test_input = problem.test_pairs[0].x
    examples_text = '\n'.join(examples)
    
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
    
    return prompt

def main_grpo_standard():
    """Main GRPO training with standard TRL"""
    
    logger.info("🚀 Starting GPT-OSS ARC GRPO Training (Standard TRL)")
    logger.info("="*60)
    
    try:
        # Setup WandB
        wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
        wandb.init(
            project="gpt-oss-arc-grpo-standard",
            name=f"grpo_std_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        
        # Load model and tokenizer with BitsAndBytesConfig
        logger.info("Loading model and tokenizer...")
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/gpt-oss-20b-BF16", 
            trust_remote_code=True,
            padding_side="left",
        )
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/gpt-oss-20b-BF16",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # Add LoRA adapters
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Memory optimization
        torch.cuda.empty_cache()
        
        logger.info("✅ Model and tokenizer loaded successfully")
        
        # Prepare dataset - Start with just 1 problem for memory testing
        problems = train_problems[:1]  # Just 1 problem for testing
        logger.info(f"📊 Loaded {len(problems)} ARC problems")
        
        # Create prompts
        prompts = []
        targets = []
        
        for problem in problems:
            prompt = create_arc_prompt(problem)
            prompts.append(prompt)
            targets.append(problem.test_pairs[0].y)
        
        # Create dataset
        dataset_dict = {
            "prompt": prompts,  # GRPO expects 'prompt' field
            "problem_id": [p.uid for p in problems]
        }
        
        train_dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"📊 Created dataset with {len(train_dataset)} examples")
        
        # GRPO Config - Memory optimized
        grpo_config = GRPOConfig(
            output_dir=str(DATA_DIR / "checkpoints_grpo_standard"),
            learning_rate=1e-5,
            num_train_epochs=1,
            per_device_train_batch_size=1,  # Keep at 1
            gradient_accumulation_steps=1,  # Reduce from 2 to 1
            warmup_steps=2,  # Reduce warmup
            logging_steps=1,
            save_steps=5,  # More frequent saves
            bf16=True,
            fp16=False,
            optim="adamw_8bit",
            report_to="wandb",
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Reduce memory usage
            # GRPO specific - Memory optimized
            num_iterations=1,
            epsilon=0.2,
            epsilon_high=0.28,
            beta=0.0,  # Remove KL term as recommended
            loss_type="dapo",
            max_prompt_length=512,  # Reduce from 1024
            max_completion_length=256,  # Reduce from 512
            num_generations=2,
            generation_batch_size=2,  # Must be divisible by num_generations
            temperature=0.7,
            top_p=0.9,
            max_steps=10,  # Reduce for testing
        )
        
        # Custom reward function - accept any combination of parameters
        def reward_function(*args, **kwargs) -> List[float]:
            """Compute rewards for GRPO - flexible parameter handling"""
            
            # Extract responses from args/kwargs
            responses = None
            if 'responses' in kwargs:
                responses = kwargs['responses']
            elif 'completions' in kwargs:
                responses = kwargs['completions']
            elif len(args) >= 3:
                responses = args[2]  # Third argument is usually responses
            elif len(args) >= 1:
                responses = args[0]  # Fallback to first argument
            
            if responses is None:
                logger.error("Could not find responses in reward function arguments")
                return [0.0]
            
            rewards = []
            logger.info(f"Computing rewards for {len(responses)} responses")
            
            for i, response in enumerate(responses):
                # Get corresponding target
                target_idx = i % len(targets)
                target_grid = targets[target_idx]
                
                # Compute reward
                reward = compute_arc_reward(response, target_grid)
                rewards.append(reward)
                
                logger.info(f"Response {i}: reward = {reward:.2f}")
            
            return rewards
        
        # Create trainer - TRL GRPOTrainer interface
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_function,  # Use reward_funcs (plural) instead of reward_function
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        
        logger.info("🚀 Starting GRPO training...")
        
        # Train
        trainer.train()
        
        logger.info("✅ Training completed!")
        
        # Save model
        model.save_pretrained(DATA_DIR / "final_model_grpo_standard")
        tokenizer.save_pretrained(DATA_DIR / "final_model_grpo_standard")
        logger.info(f"💾 Model saved to {DATA_DIR / 'final_model_grpo_standard'}")
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        wandb.finish()

if __name__ == "__main__":
    main_grpo_standard()