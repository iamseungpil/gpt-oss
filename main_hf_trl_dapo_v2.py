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
os.environ["WANDB_API_KEY"] = "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"
# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 🔥 Port collision fix
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "39503"
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
    StoppingCriteria,
    StoppingCriteriaList,
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
DATA_DIR = Path("/data/gpt_oss_final")
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


def create_harmony_prompt(problem) -> str:
    """Create proper Harmony format prompt for ARC problem with 30k token support."""
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(
            f"Example {i}:\n"
            f"Input:\n{grid_to_string(train_pair.x)}\n"
            f"Output:\n{grid_to_string(train_pair.y)}"
        )
    
    test_input = problem.test_pairs[0].x
    examples_text = '\n\n'.join(examples)
    
    # Updated prompt to encourage longer reasoning for 30k tokens
    prompt = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># ARC Puzzle Solver

You are an expert at solving Abstract Reasoning Corpus (ARC) puzzles.

## Instructions
You MUST use channels to structure your response:
1. Use <|channel|>analysis<|message|> for detailed pattern identification and reasoning
   - Examine each training example carefully
   - Identify transformation rules step by step
   - Be thorough and comprehensive in your analysis
2. Use <|channel|>final<|message|> for providing the final solution grid

## Important
- Take your time to analyze the patterns thoroughly
- Your analysis should be detailed and comprehensive
- Only switch to final channel when you're confident in your solution

## Example Response Format
<|channel|>analysis<|message|>
Looking at the training examples, I need to analyze each one carefully...
[extensive pattern analysis with multiple paragraphs]
[detailed reasoning about the transformation rules]
[step-by-step application to test input]
<|channel|>final<|message|>
[solution grid with numbers only]<|end|><|start|>user<|message|>Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

Analyze the pattern thoroughly, then provide the solution grid.<|end|><|start|>assistant<|channel|>"""
    
    return prompt


def compute_four_component_reward(response: str, target_grid: np.ndarray, use_final_channel: bool = True, current_step: int = 0) -> Dict[str, float]:
    """
    Compute 4-component reward with curriculum learning:
    1. Format reward: Can extract valid grid
    2. Size accuracy: Grid has correct shape
    3. Pixel accuracy: Cell-wise correctness
    4. Final channel penalty: Proper format usage (curriculum learning)
    """
    
    # Initialize components
    format_reward = 0.0
    size_accuracy = 0.0 
    pixel_accuracy = 0.0
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
    
    # 1. Format reward: Can we extract a valid grid?
    predicted_grid = parse_grid_from_response(predicted_text, target_grid.shape)
    if predicted_grid is not None:
        format_reward = 0.5
        
        # 2. Size accuracy
        if predicted_grid.shape == target_grid.shape:
            size_accuracy = 0.5
            
            # 3. Pixel accuracy
            correct_pixels = np.sum(predicted_grid == target_grid)
            total_pixels = target_grid.size
            pixel_accuracy = (correct_pixels / total_pixels) * 2.0
            
            # Perfect grid bonus
            if correct_pixels == total_pixels:
                pixel_accuracy += 0.5
        else:
            size_accuracy = -0.1
    else:
        format_penalty_strength = min(0.1 + (current_step * 0.01), 0.5)
        format_reward = -format_penalty_strength
    
    # Calculate total reward
    total_reward = format_reward + size_accuracy + pixel_accuracy - final_channel_penalty
    
    # Clear any temporary variables to prevent memory buildup
    del predicted_text
    if predicted_grid is not None:
        del predicted_grid
    
    return {
        'format_reward': format_reward,
        'size_accuracy': size_accuracy,
        'pixel_accuracy': pixel_accuracy,
        'final_channel_penalty': final_channel_penalty,
        'total_reward': total_reward
    }


def main():
    logger.info("=" * 80)
    logger.info("🚀 Starting GPT-OSS ARC Training with HF TRL DAPO v2")
    logger.info("📊 30,000 token support + Memory optimization")
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
            "reward_components": 4,
            "dataset_size": 400,
        }
    )
    
    # Model configuration with 8-bit quantization
    model_name = "openai/gpt-oss-20b"
    logger.info(f"📦 Loading model: {model_name}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
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
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Clear initial memory
    clear_memory_cache()
    
    # Prepare dataset
    logger.info("📊 Preparing dataset...")
    prompts = []
    all_targets = []
    
    problems = train_problems[:400]
    for problem in problems:
        prompt = create_harmony_prompt(problem)
        prompts.append(prompt)
        all_targets.append(problem.test_pairs[0].y)
    
    # Create dataset
    dataset_dict = {
        "prompt": prompts,
        "problem_id": [p.uid for p in problems]
    }
    
    train_dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"📊 Created dataset with {len(train_dataset)} examples")
    
    # GRPO Config with 30k token support
    grpo_config = GRPOConfig(
        output_dir=str(DATA_DIR / "checkpoints_hf_trl_dapo_v2"),
        learning_rate=5e-6,  # Reduced for stability with long sequences
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Accumulate for effective batch size
        warmup_steps=10,
        logging_steps=1,
        save_steps=25,
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
        max_completion_length=30000,  # Support 30k tokens
        num_generations=2,
        generation_batch_size=1,  # Reduced to prevent OOM with 30k tokens
        max_steps=200,  # More training steps
        
        # Memory optimization
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )
    
    # Create reward function that tracks all targets
    current_step = [0]  # Mutable counter for curriculum learning
    
    def reward_function(completions: List[str], prompts: List[str]) -> List[Dict[str, float]]:
        """Compute rewards for batch of completions."""
        rewards = []
        for i, (completion, prompt) in enumerate(zip(completions, prompts)):
            # Find corresponding target grid
            prompt_idx = prompts.index(prompt) if prompt in prompts else 0
            target_grid = all_targets[prompt_idx % len(all_targets)]
            
            # Compute reward
            reward = compute_four_component_reward(
                completion, 
                target_grid,
                use_final_channel=True,
                current_step=current_step[0]
            )
            rewards.append(reward)
            
            # Log rewards
            if i == 0:  # Log first example
                logger.info(f"Step {current_step[0]} Reward: {reward}")
        
        current_step[0] += 1
        
        # Periodic memory cleanup
        if current_step[0] % 10 == 0:
            clear_memory_cache()
        
        return rewards
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Build stopping criteria for 30k tokens
    stop_id_sequences: List[List[int]] = []
    for stop_str in harmony_stop_tokens:
        tokens = tokenizer(stop_str, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
        if tokens:
            stop_id_sequences.append(tokens)
    
    stopping_criteria = StoppingCriteriaList([StopOnSequences(stop_id_sequences)])
    
    # Configure generation kwargs for 30k tokens
    generation_kwargs = {
        "max_new_tokens": 30000,  # 30k tokens
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "stopping_criteria": stopping_criteria,
    }
    
    trainer.generation_kwargs = generation_kwargs
    
    try:
        logger.info("🎯 Starting training...")
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        final_model_path = DATA_DIR / "final_model_hf_trl_dapo_v2"
        trainer.save_model(str(final_model_path))
        
        logger.info(f"✅ Training complete! Model saved to {final_model_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        clear_memory_cache()
        wandb.finish()


if __name__ == "__main__":
    main()