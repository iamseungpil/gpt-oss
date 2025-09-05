#!/usr/bin/env python3
"""
GPT-OSS 20B ARC Training with HuggingFace TRL + QLoRA DAPO (v2)
================================================================
Updated version with:
1. 30,000 token support for proper channel switching
2. Memory leak prevention
3. Optimized for long generation sequences
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
import gc

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Respect existing WANDB_API_KEY if set; do not overwrite secrets in runtime
os.environ.setdefault("WANDB_API_KEY", os.environ.get("WANDB_API_KEY", ""))
# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# CUDA_VISIBLE_DEVICES should be set in shell before running this script

# 🔥 Port collision fix
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "39503"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"

# 🕒 Increase timeouts for 30k token generation
os.environ["NCCL_TIMEOUT"] = "7200"  # 2 hours
os.environ["TORCH_DISTRIBUTED_DETAIL"] = "DEBUG"
os.environ["TIMEOUT"] = "7200"  # 2 hours general timeout

# Core imports
import wandb
from arc import train_problems
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F

# Import openai-harmony for proper GPT-OSS formatting
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    SystemContent,
    DeveloperContent,
    TextContent,
    StreamableParser
)

# Configuration
DATA_DIR = Path("/opt/dlami/nvme/gpt_oss")
LOG_FILE = DATA_DIR / "logs" / "hf_trl_dapo_v2.log"
RESPONSE_DIR = DATA_DIR / "logs" / "hf_trl_responses_v2"

# Initialize Harmony encoding for GPT-OSS
harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
harmony_stop_tokens = harmony_encoding.stop_tokens_for_assistant_actions()


class StopOnSequences(StoppingCriteria):
    """Stop generation when any of the provided token-id sequences appears."""

    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_sequences:
            return False
        generated = input_ids[0].tolist()
        for seq in self.stop_sequences:
            L = len(seq)
            if L <= len(generated) and generated[-L:] == seq:
                return True
        return False


def clear_memory_cache():
    """Clear GPU and CPU memory cache to prevent memory leaks."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


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


def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format."""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)


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


def create_harmony_prompt(problem, tokenizer=None) -> list:
    """Create proper harmony format messages for ARC problem."""
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(
            f"Example {i}:\n"
            f"Input:\n{grid_to_string(train_pair.x)}\n"
            f"Output:\n{grid_to_string(train_pair.y)}"
        )
    
    test_input = problem.test_pairs[0].x
    examples_text = '\n\n'.join(examples)
    
    # Return messages in proper format (same as inference)
    messages = [
        {
            "role": "system", 
            "content": """You are ChatGPT, a large language model trained by OpenAI.

Reasoning: medium

# Valid channels: analysis, commentary, final"""
        },
        {
            "role": "user", 
            "content": f"""# ARC Puzzle Solver

You are solving Abstract Reasoning Corpus (ARC) puzzles.

For each puzzle:
1. Use the analysis channel for examining patterns and reasoning
2. Identify the transformation rule from training examples
3. Apply the rule to the test input
4. Switch to the final channel for your solution grid

You will naturally switch channels as you progress through the solution.

## Task
Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

What is the output grid?"""
        }
    ]
    
    return messages


def compute_five_component_reward(response: str, target_grid: np.ndarray, use_final_channel: bool = True, current_step: int = 0) -> Dict[str, float]:
    """
    Compute 5-component reward with curriculum learning:
    1. Format reward: Can extract valid grid
    2. Size accuracy: Grid has correct shape  
    3. Pixel accuracy: Cell-wise correctness
    4. Length reward: Encourages detailed reasoning
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
    
    # FINAL CHANNEL PENALTY REMOVED - Format penalty is sufficient
    final_channel_penalty = 0.0
    
    # 4. Length reward: Encourage detailed reasoning (analysis channel)
    analysis_start = predicted_text.find("<|channel|>analysis<|message|>")
    final_start = predicted_text.find("<|channel|>final<|message|>")
    
    if analysis_start != -1 and final_start != -1:
        analysis_length = final_start - analysis_start
        
        # Improved length reward with penalty for excessive length (20,000+ chars)
        if analysis_length > 20000:  # Excessive length - penalty
            length_reward = max(-0.2, 0.3 - (analysis_length - 20000) / 30000)
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
    
    # Calculate total reward (final_channel_penalty removed)
    total_reward = format_reward + size_accuracy + pixel_accuracy + length_reward
    
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
    logger.info("🚀 Starting GPT-OSS ARC Continual Learning with DAPO v2")
    logger.info("📊 30,000 token support + Sequential problem training")
    logger.info("=" * 80)
    
    # Initialize wandb
    wandb.init(
        project="gpt-oss-arc-dapo-v2",
        name=f"hf_trl_dapo_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": "openai/gpt-oss-20b",
            "method": "HF TRL GRPO/DAPO",
            "max_tokens": 30000,
            "memory_optimized": True,
            "reward_components": 5,
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
    tokenizer.padding_side = "left"
    
    # Load model with existing quantization (GPU 6 via CUDA_VISIBLE_DEVICES)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Will use GPU 0 (which is actually GPU 6)
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
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
    
    # Apply LoRA and ensure dtype consistency with base model (bf16)
    model = get_peft_model(model, lora_config)
    try:
        model = model.to(dtype=torch.bfloat16)
    except Exception:
        pass
    model.print_trainable_parameters()
    
    # Clear initial memory
    clear_memory_cache()
    
    # Load all 10 problems for continual learning
    # Allow overriding number of problems via env (default: 10)
    num_problems = int(os.environ.get("NUM_PROBLEMS", "10"))
    problems = train_problems[:num_problems]  # Problems 0..num_problems-1
    logger.info(f"📊 Loaded {len(problems)} problems for continual learning")
    
    # Continual learning: train each problem sequentially
    for problem_idx, problem in enumerate(problems):
        logger.info("=" * 60)
        logger.info(f"🎯 CONTINUAL LEARNING: Problem {problem_idx + 1}/10")
        logger.info(f"📋 Problem UID: {problem.uid}")
        logger.info("=" * 60)
        
        # Create single-problem dataset using messages format
        messages = create_harmony_prompt(problem, tokenizer)
        # Convert messages to prompt using tokenizer.apply_chat_template
        # Rely on chat template to add the correct assistant generation prefix
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="medium"
        )
        dataset_dict = {
            "prompt": [prompt],
            "problem_id": [problem.uid]
        }
        
        train_dataset = Dataset.from_dict(dataset_dict)
        all_targets = [problem.test_pairs[0].y]
    
        # GRPO Config with 30k token support
        # Optional overrides via env for quick smoke tests
        max_steps_override = int(os.environ.get("MAX_STEPS", "50"))
        save_steps_override = int(os.environ.get("SAVE_STEPS", "25"))

        max_completion_override = int(os.environ.get("MAX_COMPLETION_LENGTH", "12000"))
        num_generations_override = int(os.environ.get("NUM_GENERATIONS", "2"))
        gen_batch_override = int(os.environ.get("GENERATION_BATCH_SIZE", str(num_generations_override)))

        grpo_config = GRPOConfig(
            output_dir=str(DATA_DIR / "checkpoints_hf_trl_dapo_v2"),
            learning_rate=5e-6,  # Reduced for stability with long sequences
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # Further increased for 30k token sequences
            warmup_steps=5,
            logging_steps=1,
            save_steps=save_steps_override,
            bf16=True,
            fp16=False,
            optim="adamw_torch",
            report_to="wandb",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            
            # GRPO/DAPO specific parameters
            num_iterations=1,
            epsilon=0.2,
            epsilon_high=0.28,
            beta=0.0,
            loss_type="bnpo",
            max_prompt_length=4096,  # Increased for ARC prompts
            max_completion_length=max_completion_override,
            num_generations=num_generations_override,  # GRPO requires at least 2 generations
            generation_batch_size=gen_batch_override,  # Match num_generations
            max_steps=max_steps_override,  # steps per problem
            
            # Memory optimization
            gradient_checkpointing=True,
            max_grad_norm=1.0,
        )
        
        # Create reward function for this specific problem  
        current_step = [0]  # Reset for each problem
        
        def reward_function(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
            """Compute rewards for batch of completions."""
            rewards = []
            for i, (completion, prompt) in enumerate(zip(completions, prompts)):
                target_grid = all_targets[0]  # Single problem target
                
                # Compute reward
                reward_dict = compute_five_component_reward(
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
            
            current_step[0] += 1
            
            # Frequent memory cleanup to prevent 130GB buildup
            if current_step[0] % 5 == 0:
                clear_memory_cache()
                import gc
                gc.collect()
            
            return rewards
        
        # Create trainer for this problem
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_function,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        
        # GRPO handles generation with the configured limits
        
        try:
            logger.info(f"🎯 Starting training for problem {problem_idx + 1}...")
            trainer.train()
            
            # Save model after each problem
            problem_model_path = DATA_DIR / f"model_after_problem_{problem_idx + 1}"
            trainer.save_model(str(problem_model_path))
            logger.info(f"💾 Model saved after problem {problem_idx + 1}: {problem_model_path}")
            
        except Exception as e:
            logger.error(f"❌ Training failed for problem {problem_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            break  # Stop continual learning if one problem fails
        
        # Clear memory between problems
        clear_memory_cache()
    
    # Save final model
    logger.info("💾 Saving final continual learning model...")
    final_model_path = DATA_DIR / "final_continual_model_hf_trl_dapo_v2"
    trainer.save_model(str(final_model_path))
    logger.info(f"✅ Continual learning complete! Final model saved to {final_model_path}")
    
    # Cleanup
    clear_memory_cache()
    wandb.finish()


if __name__ == "__main__":
    continual_learning_main()
