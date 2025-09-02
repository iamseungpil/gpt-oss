#!/usr/bin/env python3
"""
GPT-OSS ARC DAPO Training v3 with HuggingFace TRL
- Improved length penalty for overly long responses
- Enhanced reward system with better balance
- Same overall structure as v2 but with refined length handling
"""

import os
import gc
import torch
import wandb
import logging
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Memory management
torch.cuda.empty_cache()
gc.collect()

# Environment variables
os.environ["WANDB_PROJECT"] = "gpt-oss-arc-dapo-v3"
os.environ["WANDB_API_KEY"] = "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"
# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# CUDA_VISIBLE_DEVICES should be set in shell before running this script

# 🔥 Port collision fix
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "39504"  # Different port from v2

# Distributed training fix
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"

# Core imports
import wandb
from arc import train_problems
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def clear_memory_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def grid_to_string(grid: np.ndarray) -> str:
    """Convert grid to string representation."""
    return '\n'.join(' '.join(map(str, row)) for row in grid)

def string_to_grid(grid_str: str) -> np.ndarray:
    """Convert string grid back to numpy array."""
    try:
        lines = grid_str.strip().split('\n')
        grid = []
        for line in lines:
            if line.strip():
                row = [int(x) for x in line.split()]
                if row:
                    grid.append(row)
        return np.array(grid)
    except Exception as e:
        logger.error(f"❌ Error parsing grid: {e}")
        return None

def parse_grid_from_response(response: str, expected_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Extract grid from model response - handles various formats."""
    if not response:
        return None
    
    # Look for final channel content first
    final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)'
    final_match = re.search(final_pattern, response, re.DOTALL)
    
    if final_match:
        content = final_match.group(1)
    else:
        content = response
    
    # Try to extract numbers in grid format
    lines = content.strip().split('\n')
    grid = []
    
    for line in lines:
        # Extract numbers from line
        numbers = re.findall(r'\d+', line)
        if numbers:
            row = [int(n) for n in numbers]
            if row:
                grid.append(row)
    
    if not grid:
        return None
    
    # Convert to numpy array
    try:
        grid_array = np.array(grid)
        
        # Validate shape if expected
        if expected_shape and grid_array.shape != expected_shape:
            # Try to find a subgrid of the right shape
            if len(grid) >= expected_shape[0]:
                truncated = []
                for row in grid[:expected_shape[0]]:
                    if len(row) >= expected_shape[1]:
                        truncated.append(row[:expected_shape[1]])
                    else:
                        break
                if len(truncated) == expected_shape[0]:
                    grid_array = np.array(truncated)
        
        return grid_array
    except:
        return None

def create_harmony_prompt(problem) -> str:
    """Create Harmony format prompt for ARC problem."""
    system_msg = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.

# ARC Problem Solving Instructions

## Goal
Analyze the Abstract Reasoning Corpus (ARC) puzzle and determine the transformation pattern to solve the test input.

## Response Format Requirements
You MUST use this exact channel format:

1. Use <|channel|>analysis<|message|> for your reasoning process:
   - Analyze each training example step by step
   - Identify the transformation pattern
   - Apply it to the test input

2. Use <|channel|>final<|message|> for providing the final solution grid

## Analysis Guidelines
- Be thorough but concise (aim for 1000-2500 characters in analysis)
- Focus on pattern identification and logical reasoning
- Avoid overly verbose explanations (over 3000 characters will be penalized)

Reasoning: medium<|end|><|start|>user<|message|>Solve this ARC puzzle by first analyzing the pattern, then providing the solution.

## Training Examples

"""
    
    # Add training examples
    examples_text = ""
    for i, pair in enumerate(problem.train_pairs):
        input_grid = grid_to_string(pair.x)
        output_grid = grid_to_string(pair.y)
        examples_text += f"### Example {i+1}:\nInput:\n{input_grid}\n\nOutput:\n{output_grid}\n\n"
    
    # Test input
    test_input = problem.test_pairs[0].x
    
    # Final prompt construction
    prompt = f"""{system_msg}{examples_text}

## Instructions
1. Use <|channel|>analysis<|message|> to:
   - Analyze what changes from input to output in each example
   - Identify the consistent transformation pattern  
   - Apply it to the test input
2. Use <|channel|>final<|message|> for providing the final solution grid

## Example Response Format
<|channel|>analysis<|message|>
Looking at the training examples, I can see that...
[pattern analysis and reasoning]
[application to test input]
<|channel|>final<|message|>
[solution grid with numbers only]<|end|><|start|>user<|message|>Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

Analyze the pattern thoroughly, then provide the solution grid.<|end|><|start|>assistant<|channel|>"""
    
    return prompt


def compute_five_component_reward_v3(response: str, target_grid: np.ndarray, use_final_channel: bool = True, current_step: int = 0) -> Dict[str, float]:
    """
    Compute 5-component reward with curriculum learning and improved length penalty:
    1. Format reward: Can extract valid grid
    2. Size accuracy: Grid has correct shape  
    3. Pixel accuracy: Cell-wise correctness
    4. Length reward: Balanced reasoning length (with penalty for excessive length)
    5. Final channel penalty: Proper format usage (curriculum learning)
    """
    
    # Initialize components
    format_reward = 0.0
    size_accuracy = 0.0 
    pixel_accuracy = 0.0
    length_reward = 0.0
    final_channel_penalty = 0.0
    
    # Extract text from response
    if isinstance(response, str):
        predicted_text = response
    else:
        predicted_text = str(response)
    
    # CURRICULUM LEARNING: Gradually increase final channel penalty
    if use_final_channel:
        has_final = (
            "<|channel|>final<|message|>" in predicted_text or
            "<|channel|>final<|" in predicted_text or
            "channel|>final" in predicted_text
        )
        if not has_final:
            penalty_strength = min(0.05 + (current_step * 0.01), 0.5)
            final_channel_penalty = penalty_strength
    
    # 4. IMPROVED Length reward with penalty for excessive length
    analysis_start = predicted_text.find("<|channel|>analysis<|message|>")
    final_start = predicted_text.find("<|channel|>final<|message|>")
    
    if analysis_start != -1 and final_start != -1:
        analysis_length = final_start - analysis_start
        
        # Improved length reward with penalties for too long responses
        if analysis_length > 4000:  # Excessive length - strong penalty
            length_reward = max(-0.2, 0.3 - (analysis_length - 4000) / 10000)
        elif analysis_length > 3000:  # Too long - moderate penalty  
            length_reward = max(0, 0.3 - (analysis_length - 3000) / 15000)
        elif analysis_length >= 1000:  # Good length - reward
            length_reward = min(0.3, analysis_length / 10000)
        elif analysis_length >= 500:  # Acceptable length - small reward
            length_reward = 0.1
        else:  # Too short - no reward
            length_reward = 0
    
    # 1. Format reward: Can we extract a valid grid?
    predicted_grid = parse_grid_from_response(predicted_text, target_grid.shape)
    if predicted_grid is not None:
        format_reward = 0.4  # Reduced from 0.5
        
        # 2. Size accuracy
        if predicted_grid.shape == target_grid.shape:
            size_accuracy = 0.4  # Reduced from 0.5
            
            # 3. Pixel accuracy (reduced weight)
            correct_pixels = np.sum(predicted_grid == target_grid)
            total_pixels = target_grid.size
            pixel_accuracy = (correct_pixels / total_pixels) * 1.5  # Reduced from 2.0
            
            # Perfect grid bonus
            if correct_pixels == total_pixels:
                pixel_accuracy += 0.3  # Reduced from 0.5
        else:
            size_accuracy = -0.1
    else:
        format_penalty_strength = min(0.1 + (current_step * 0.01), 0.5)
        format_reward = -format_penalty_strength
    
    # Calculate total reward
    total_reward = format_reward + size_accuracy + pixel_accuracy + length_reward - final_channel_penalty
    
    # Clear any temporary variables to prevent memory buildup
    del predicted_text
    if predicted_grid is not None:
        del predicted_grid
    
    return {
        'format_reward': format_reward,
        'size_accuracy': size_accuracy,
        'pixel_accuracy': pixel_accuracy,
        'length_reward': length_reward,
        'final_channel_penalty': final_channel_penalty,
        'total_reward': total_reward
    }


def continual_learning_main():
    logger.info("=" * 80)
    logger.info("🚀 Starting GPT-OSS ARC Continual Learning with DAPO v3")
    logger.info("📊 Improved length penalty + Sequential problem training")
    logger.info("=" * 80)
    
    # Initialize wandb
    wandb.init(
        project="gpt-oss-arc-dapo-v3",
        name=f"hf_trl_dapo_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": "openai/gpt-oss-20b",
            "method": "HF TRL GRPO/DAPO v3",
            "max_tokens": 8000,
            "memory_optimized": True,
            "reward_components": 5,
            "length_penalty_improved": True,
            "dataset_size": 10,
            "continual_learning": True,
            "steps_per_problem": 50,
        }
    )
    
    # Model configuration - use model's existing quantization
    model_name = "openai/gpt-oss-20b"
    logger.info(f"📦 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with existing quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Clear initial memory
    clear_memory_cache()
    
    # Load all 10 problems for continual learning
    problems = train_problems[:10]  # Problems 0-9
    logger.info(f"📊 Loaded {len(problems)} problems for continual learning")
    
    # Continual learning: train each problem sequentially
    for problem_idx, problem in enumerate(problems):
        logger.info("=" * 60)
        logger.info(f"🎯 CONTINUAL LEARNING: Problem {problem_idx + 1}/10")
        logger.info(f"📋 Problem UID: {problem.uid}")
        logger.info("=" * 60)
        
        # Create single-problem dataset
        prompt = create_harmony_prompt(problem)
        dataset_dict = {
            "prompt": [prompt],
            "problem_id": [problem.uid]
        }
        dataset = Dataset.from_dict(dataset_dict)
        
        # Prepare targets for reward calculation  
        all_targets = [problem.test_pairs[0].y]
        
        # Create reward function for this specific problem  
        current_step = [0]  # Reset for each problem
        
        def reward_function(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
            """Compute rewards for batch of completions."""
            rewards = []
            for i, (completion, prompt) in enumerate(zip(completions, prompts)):
                target_grid = all_targets[0]  # Single problem target
                
                # Compute reward using v3 function
                reward_dict = compute_five_component_reward_v3(
                    completion, 
                    target_grid,
                    use_final_channel=True,
                    current_step=current_step[0]
                )
                rewards.append(reward_dict['total_reward'])  # Return only the float value
                
                # Log rewards
                if i == 0:  # Log first example
                    logger.info(f"Problem {problem_idx + 1} Step {current_step[0]} Reward: {reward_dict['total_reward']:.3f}")
                    logger.info(f"  Components - Format: {reward_dict['format_reward']:.2f}, Size: {reward_dict['size_accuracy']:.2f}, Pixel: {reward_dict['pixel_accuracy']:.2f}")
                    logger.info(f"  Length: {reward_dict['length_reward']:.2f}, Final Channel Penalty: {reward_dict['final_channel_penalty']:.2f}")
            
            current_step[0] += 1
            
            # Periodic memory cleanup
            if current_step[0] % 10 == 0:
                clear_memory_cache()
            
            return rewards
        
        # Create trainer for this problem
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_function,
            tokenizer=tokenizer,
            args=GRPOConfig(
                output_dir=f"/opt/dlami/nvme/gpt_oss/problem_{problem_idx}",
                num_train_epochs=50,
                max_completion_length=8000,  # Reduced for faster training
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                dataloader_drop_last=True,
                eval_strategy="no",
                save_strategy="no",
                logging_steps=1,
                remove_unused_columns=False,
                gradient_checkpointing=True,
                warmup_steps=5,
                max_grad_norm=1.0,
                learning_rate=5e-6,
                lr_scheduler_type="cosine",
                report_to="wandb"
            ),
            train_dataset=dataset,
        )
        
        logger.info(f"🎯 Starting training for problem {problem_idx + 1}...")
        
        # Train for 50 steps on this problem
        trainer.train()
        
        logger.info(f"✅ Completed problem {problem_idx + 1}/10")
        
        # Memory cleanup between problems
        clear_memory_cache()
    
    # Save final model
    logger.info("💾 Saving final continual learning model...")
    final_model_path = "/opt/dlami/nvme/gpt_oss/final_continual_model_hf_trl_dapo_v3"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"✅ Continual learning complete! Final model saved to {final_model_path}")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    continual_learning_main()