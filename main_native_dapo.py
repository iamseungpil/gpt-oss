#!/usr/bin/env python3
"""
GPT-OSS 20B ARC Training with Native Unsloth DAPO
==============================================

Using Unsloth's built-in DAPO implementation for more efficient and stable training.
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

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["WANDB_API_KEY"] = "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"
# Disable torch compile completely
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.disable = True

# Core imports
import wandb
from arc import train_problems
from datasets import Dataset

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Model settings
    MODEL_NAME = "unsloth/gpt-oss-20b-BF16"
    MAX_SEQ_LENGTH = 2048
    DTYPE = torch.bfloat16
    
    # Training settings
    MAX_STEPS = 500
    BATCH_SIZE = 1
    LEARNING_RATE = 2e-6
    GRADIENT_ACCUMULATION_STEPS = 2
    
    # DAPO settings
    LOSS_TYPE = "dapo"
    EPSILON = 0.2
    EPSILON_HIGH = 0.28
    DELTA = 1.5
    
    # Early stopping
    PERFECT_THRESHOLD = 1.0
    MIN_PERFECT_SOLUTIONS = 3
    
    # Paths
    DATA_DIR = Path("/data/gpt_oss_final")
    LOG_FILE = DATA_DIR / "logs" / "native_dapo.log"
    RESPONSES_DIR = DATA_DIR / "logs" / "dapo_responses"
    CHECKPOINT_DIR = DATA_DIR / "checkpoints"
    FINAL_MODEL_DIR = DATA_DIR / "final_model"

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Setup logging configuration"""
    Config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG to see reward details
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# GRID UTILITIES
# =============================================================================

def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format"""
    return '\\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

def string_to_grid(text: str) -> Optional[np.ndarray]:
    """Extract grid ONLY from final channel - strict mode"""
    try:
        # Only extract from final channel (strict harmony format)
        if "<|channel|>final<|" in text:
            parts = text.split("<|channel|>final<|")
            if len(parts) > 1:
                final_content = parts[1]
                # Find the message content
                if "message|>" in final_content:
                    final_content = final_content.split("message|>", 1)[1]
                # Stop at end marker
                for marker in ["<|end|>", "<|channel|>", "<|return|>"]:
                    if marker in final_content:
                        final_content = final_content.split(marker)[0]
                text = final_content.strip()
                
                # Look for code blocks first (```...```)
                if "```" in text:
                    code_blocks = text.split("```")
                    for block in code_blocks:
                        if block.strip():
                            # Try to parse this block as a grid
                            grid = parse_grid_block(block.strip())
                            if grid is not None:
                                return grid
                
                # If no code blocks, parse the entire text
                return parse_grid_block(text)
        
        # NO final channel = NO grid extraction (strict mode)
        return None
                
    except Exception as e:
        logger.debug(f"Grid parsing error: {e}")
    
    return None

def parse_grid_block(text: str) -> Optional[np.ndarray]:
    """Parse a text block to extract grid"""
    lines = text.strip().split('\n')
    grid_rows = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip header lines
        if any(skip in line.lower() for skip in ['grid:', 'output:', 'answer:', 'result:', 'input:', 'example', 'test case', 'applying', 'resulting']):
            continue
        
        # Skip lines with explanatory text
        if any(word in line.lower() for word in ['the', 'pattern', 'rule', 'transform', 'analysis', 'solution', 'task', 'note']):
            continue
            
        # Process potential grid line
        parts = line.split()
        if parts:
            try:
                row = []
                for part in parts:
                    # Handle decimal numbers by converting to int
                    if '.' in part:
                        num = int(float(part))
                    else:
                        num = int(part)
                    
                    # Only accept digits 0-9
                    if 0 <= num <= 9:
                        row.append(num)
                    else:
                        break  # Invalid number, skip this line
                
                # Only add if we processed all parts successfully
                if len(row) == len(parts) and len(row) > 0:
                    grid_rows.append(row)
                elif grid_rows:
                    # Stop if we hit invalid line after starting grid
                    break
                    
            except (ValueError, OverflowError):
                if grid_rows:
                    break
                continue
    
    if grid_rows:
        row_lengths = [len(row) for row in grid_rows]
        if len(set(row_lengths)) == 1 and len(grid_rows) >= 2:
            return np.array(grid_rows)
    
    return None

# =============================================================================
# PROMPT CREATION
# =============================================================================

def create_arc_prompt(problem) -> Tuple[str, np.ndarray]:
    """Create ARC prompt using proper Harmony format with system prompt"""
    
    # Build examples section
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(f"Example {i}: Input:\\n{grid_to_string(train_pair.x)}\\nOutput:\\n{grid_to_string(train_pair.y)}\\n")
    
    # Test section
    test_input = problem.test_pairs[0].x
    test_output = problem.test_pairs[0].y
    examples_text = '\\n'.join(examples)
    
    # Use proper Harmony format with reasoning setting
    prompt = f"""<|start|>system<|message|>You are a pattern solver. Use high reasoning.
Reasoning: high
Output format: Provide the final answer as a grid of numbers.<|end|>
<|start|>user<|message|>Solve this pattern and output the grid:
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|channel|>analysis<|message|>Let me analyze this pattern step by step.<|end|>
<|start|>assistant<|channel|>final<|message|>Based on the pattern, here is the output grid:
"""
    
    return prompt, test_output

# =============================================================================
# DATASET PREPARATION
# =============================================================================

def create_arc_dataset(problems: List) -> Dataset:
    """Create dataset for GRPO training"""
    
    data = []
    for problem in problems:
        prompt, target_grid = create_arc_prompt(problem)
        
        data.append({
            "prompt": prompt,
            "problem_id": problem.uid,
            "target_grid": target_grid.tolist()
        })
    
    return Dataset.from_list(data)

# =============================================================================
# RESPONSE SAVING
# =============================================================================

def save_responses(step: int, prompts: List[str], completions: List[str], target_grids: Optional[List]) -> None:
    """Save all responses from current step for analysis"""
    try:
        Config.RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create response data
        response_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "num_responses": len(completions),
            "responses": []
        }
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            # Extract grid if possible
            predicted_grid = string_to_grid(completion)
            has_grid = predicted_grid is not None
            
            response_entry = {
                "index": i,
                "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "completion": completion,
                "has_grid": has_grid,
                "grid_shape": predicted_grid.shape if has_grid else None,
                "response_length": len(completion)
            }
            
            # Add target grid info if available
            if target_grids and i < len(target_grids):
                target_grid = np.array(target_grids[i]) if isinstance(target_grids[i], list) else target_grids[i]
                response_entry["target_shape"] = target_grid.shape
                if has_grid:
                    response_entry["shape_match"] = predicted_grid.shape == target_grid.shape
                    if predicted_grid.shape == target_grid.shape:
                        accuracy = np.mean(predicted_grid == target_grid)
                        response_entry["accuracy"] = float(accuracy)
            
            response_data["responses"].append(response_entry)
        
        # Save to file
        filename = Config.RESPONSES_DIR / f"step_{step:05d}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        
        # Log summary
        grids_found = sum(1 for r in response_data["responses"] if r["has_grid"])
        logger.info(f"   💾 Saved {len(completions)} responses | {grids_found} grids extracted")
        
        # Log to WandB if available
        if wandb.run:
            wandb.log({
                "step": step,
                "num_grids_extracted": grids_found,
                "num_responses": len(completions),
                "avg_response_length": np.mean([r["response_length"] for r in response_data["responses"]])
            })
            
    except Exception as e:
        logger.error(f"Failed to save responses: {e}")

# =============================================================================
# REWARD FUNCTION
# =============================================================================

def compute_arc_reward(prompts, completions, trainer_state, **kwargs):
    """
    Compute comprehensive ARC reward with 4 components for GRPO.
    
    Args:
        prompts: List of prompt strings
        completions: List of completion strings
        trainer_state: Current trainer state
        **kwargs: Additional dataset columns (e.g., target_grid)
    
    Returns:
        List of reward floats
    """
    
    rewards = []
    
    # Get target grids from kwargs if available
    target_grids = kwargs.get('target_grid', None)
    
    # Save responses for analysis
    step = trainer_state.global_step if trainer_state else 0
    save_responses(step, prompts, completions, target_grids)
    
    for i, completion in enumerate(completions):
        # Initialize reward components
        format_reward = 0.0
        size_accuracy = 0.0 
        pixel_accuracy = 0.0
        final_channel_penalty = 0.0
        
        # Extract text from completion
        if isinstance(completion, str):
            predicted_text = completion
        else:
            predicted_text = str(completion)
        
        # Check if response reached final channel
        has_final = "<|channel|>final<|" in predicted_text
        if not has_final:
            # Heavy penalty for not reaching final channel
            final_channel_penalty = 1.0
            
        # 1. Format reward: Can we extract a valid grid?
        predicted_grid = string_to_grid(predicted_text)
        if predicted_grid is not None:
            format_reward = 0.2  # Base reward for extractable grid
            
            # 2. Size accuracy and pixel accuracy
            if target_grids and i < len(target_grids):
                target_grid = np.array(target_grids[i]) if isinstance(target_grids[i], list) else target_grids[i]
                
                if predicted_grid.shape == target_grid.shape:
                    size_accuracy = 0.3
                    # Calculate actual pixel accuracy
                    correct_pixels = np.sum(predicted_grid == target_grid)
                    total_pixels = target_grid.size
                    pixel_accuracy = correct_pixels / total_pixels
                else:
                    size_accuracy = -0.2
            else:
                # Default if no target grid available
                expected_shape = (3, 3)
                if predicted_grid.shape == expected_shape:
                    size_accuracy = 0.3
                    pixel_accuracy = 0.5  # Dummy accuracy
                else:
                    size_accuracy = -0.2
        else:
            format_reward = -1.0
        
        total_reward = format_reward + size_accuracy + pixel_accuracy - final_channel_penalty
        
        # Log for debugging - show stats for all completions with grids
        if predicted_grid is not None or i == 0:
            logger.debug(f"Response {i}: format={format_reward:.3f}, size={size_accuracy:.3f}, "
                        f"pixel={pixel_accuracy:.3f}, final_penalty={final_channel_penalty:.3f}, "
                        f"total={total_reward:.3f}, has_grid={predicted_grid is not None}")
        
        rewards.append(total_reward)
    
    return rewards

# =============================================================================
# MAIN TRAINING
# =============================================================================

def main_native_dapo():
    """Main DAPO training using Unsloth's native implementation"""
    
    logger.info("🚀 Starting GPT-OSS ARC Native DAPO Training")
    logger.info("=" * 50)
    
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer
        
        # Setup WandB with correct token
        wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
        wandb.init(
            project="gpt-oss-arc-dapo",
            name=f"dapo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": Config.MODEL_NAME,
                "max_steps": Config.MAX_STEPS,
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LEARNING_RATE,
                "loss_type": Config.LOSS_TYPE,
            }
        )
        logger.info("📊 WandB initialized successfully")
        
        # Load model
        logger.info(f"🚀 Loading {Config.MODEL_NAME}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=Config.MODEL_NAME,
            max_seq_length=Config.MAX_SEQ_LENGTH,
            dtype=Config.DTYPE,
            load_in_4bit=True,
            device_map={"": 0},  # Load all to GPU 0 instead of auto
            # Use Unsloth's modified template for GPT-OSS
            use_cache=False,  # Disable KV cache for training
            trust_remote_code=True  # Allow Unsloth's custom code
        )
        
        # Ensure using Unsloth's GPT-OSS chat template
        # This handles Harmony format properly
        tokenizer.chat_template = None  # Let Unsloth handle it
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=False,  # Disable gradient checkpointing to avoid shape issues
            random_state=42
        )
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✅ Model loaded | Trainable params: {trainable_params/1e6:.2f}M")
        
        # Prepare dataset
        problems = train_problems[:5]
        logger.info(f"📊 Loaded {len(problems)} ARC problems")
        for i, p in enumerate(problems):
            logger.info(f"   {i+1}. {p.uid}")
        
        train_dataset = create_arc_dataset(problems)
        logger.info(f"📊 Created dataset with {len(train_dataset)} examples")
        
        # Configure DAPO training
        training_args = GRPOConfig(
            output_dir=str(Config.CHECKPOINT_DIR),
            num_train_epochs=1,
            max_steps=Config.MAX_STEPS,
            per_device_train_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=Config.LEARNING_RATE,
            warmup_steps=50,
            logging_steps=10,
            save_steps=50,
            
            # Memory optimization
            fp16=False,  # Use bfloat16 instead
            bf16=True,
            dataloader_pin_memory=False,
            torch_compile=False,  # Disable torch compilation to fix dynamo errors
            
            # DAPO specific parameters
            loss_type=Config.LOSS_TYPE,
            epsilon=Config.EPSILON,
            epsilon_high=Config.EPSILON_HIGH,
            delta=Config.DELTA,
            mask_truncated_completions=True,
            num_generations=4,  # Reduce from default 8 to avoid memory issues
            
            # Logging
            report_to="wandb",
            run_name=f"native_dapo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        
        # Set generation config for GPT-OSS with high reasoning
        # Reduce tokens to save memory
        model.generation_config.max_new_tokens = 1024
        model.generation_config.temperature = 0.7
        model.generation_config.top_p = 0.9
        model.generation_config.do_sample = True
        
        # Set proper stop tokens for Harmony format
        stop_tokens = ["<|end|>", "<|return|>", "<|channel|>"]
        stop_token_ids = []
        for token in stop_tokens:
            if token in tokenizer.get_vocab():
                token_id = tokenizer.encode(token, add_special_tokens=False)
                if token_id:
                    stop_token_ids.extend(token_id)
        
        if stop_token_ids:
            model.generation_config.eos_token_id = stop_token_ids
        
        # Ensure proper padding
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        
        # Create trainer
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            reward_funcs=[compute_arc_reward],
        )
        
        logger.info("🚀 Starting native DAPO training...")
        logger.info(f"   Loss type: {Config.LOSS_TYPE}")
        logger.info(f"   Epsilon: {Config.EPSILON}")
        logger.info(f"   Epsilon high: {Config.EPSILON_HIGH}")
        logger.info(f"   Delta: {Config.DELTA}")
        
        # Start training
        trainer.train()
        
        logger.info("✅ Training completed!")
        
        # Save final model
        Config.FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(Config.FINAL_MODEL_DIR)
        tokenizer.save_pretrained(Config.FINAL_MODEL_DIR)
        logger.info(f"💾 Final model saved: {Config.FINAL_MODEL_DIR}")
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        # WandB disabled
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main_native_dapo()