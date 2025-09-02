#!/usr/bin/env python3
"""
GPT-OSS 20B ARC Training with HuggingFace TRL + QLoRA DAPO
==========================================================

Based on barc_post/long_with_logit_reward_dapo.py but using HuggingFace TRL
instead of Unsloth to avoid compatibility issues.
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
os.environ["WANDB_API_KEY"] = "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"
# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 🔥 Port collision fix from long_with_logit_reward_dapo.py
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "39503"  # Default port, avoids conflicts
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
LOG_FILE = DATA_DIR / "logs" / "hf_trl_dapo.log"
RESPONSE_DIR = DATA_DIR / "logs" / "hf_trl_responses"

# Initialize Harmony encoding for GPT-OSS
harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
# Get stop tokens for assistant actions (strings like <|return|>, <|end|>, etc.)
harmony_stop_tokens = harmony_encoding.stop_tokens_for_assistant_actions()


class StopOnSequences(StoppingCriteria):
    """Stop generation when any of the provided token-id sequences appears."""

    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        # Filter out empty sequences just in case
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore[override]
        if not self.stop_sequences:
            return False
        generated = input_ids[0].tolist()
        for seq in self.stop_sequences:
            L = len(seq)
            if L <= len(generated) and generated[-L:] == seq:
                return True
        return False

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

# Global variables for tracking
current_training_step = 0
all_targets = []

def parse_grid_from_response(response: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Extract grid from model response, prioritizing final channel"""
    
    # Try to extract from final channel first (harmony format)
    final_pattern = r'<\|channel\|>final<\|message\|>([\s\S]*?)(?:<\|end\|>|$)'
    final_match = re.search(final_pattern, response)
    if final_match:
        final_content = final_match.group(1).strip()
        grid = parse_grid_block(final_content, target_shape)
        if grid is not None:
            return grid
    
    # Try to find grid in code blocks
    code_block_pattern = r'```(?:python|text|)?\n?([\s\S]*?)```'
    code_blocks = re.findall(code_block_pattern, response, re.MULTILINE)
    
    for block in code_blocks:
        grid = parse_grid_block(block.strip(), target_shape)
        if grid is not None:
            return grid
    
    # Try parsing the entire response
    return parse_grid_block(response, target_shape)

def parse_grid_block(text: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Parse a text block to extract grid"""
    lines = text.strip().split('\n')
    grid_rows = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip header/comment lines
        if any(skip in line.lower() for skip in ['grid:', 'output:', 'answer:', 'result:', 'input:', 'example']):
            continue
        
        # Process potential grid line (allow comma-separated numbers)
        parts = line.replace(',', ' ').split()
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
    
    # Validate grid shape
    if grid_rows:
        row_lengths = [len(row) for row in grid_rows]
        if len(set(row_lengths)) == 1 and len(grid_rows) == target_shape[0] and row_lengths[0] == target_shape[1]:
            return np.array(grid_rows)
    
    return None

def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format"""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

def create_arc_prompt_harmony(problem) -> str:
    """Create ARC prompt with proper channel instructions
    
    Using direct string formatting since openai-harmony library
    doesn't properly render DeveloperContent text.
    """
    
    # Build examples section
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(f"Example {i}:\nInput:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}")
    
    # Test section
    test_input = problem.test_pairs[0].x
    examples_text = '\n\n'.join(examples)
    
    # Build prompt directly with proper channel format
    # Based on official GPT-OSS harmony format documentation
    prompt = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># ARC Puzzle Solver

You are an expert at solving Abstract Reasoning Corpus (ARC) puzzles.

## Instructions
You MUST use channels to structure your response:
1. Use <|channel|>analysis<|message|> for identifying patterns and reasoning
2. Use <|channel|>final<|message|> for providing the final solution grid

## Example Response Format
<|channel|>analysis<|message|>
Looking at the examples, I can see that...
[detailed pattern analysis]
<|channel|>final<|message|>
[solution grid with numbers only]<|end|><|start|>user<|message|>Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

Analyze the pattern, then provide the solution grid.<|end|><|start|>assistant<|channel|>"""
    
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
        # Check for various harmony channel formats
        has_final = (
            "<|channel|>final<|message|>" in predicted_text or
            "<|channel|>final<|" in predicted_text or
            "channel|>final" in predicted_text
        )
        if not has_final:
            # Start with very small penalty, gradually increase over more steps
            penalty_strength = min(0.05 + (current_step * 0.01), 0.5)  # 0.05 → 0.5 over 45 steps (gentler)
            final_channel_penalty = penalty_strength
    
    # 1. Format reward: Can we extract a valid grid? (IMPROVED)
    predicted_grid = parse_grid_from_response(predicted_text, target_grid.shape)
    if predicted_grid is not None:
        format_reward = 0.5  # Higher base reward for extractable grid
        
        # 2. Size accuracy
        if predicted_grid.shape == target_grid.shape:
            size_accuracy = 0.5  # Higher reward for correct shape
            
            # 3. Pixel accuracy (with bonus for perfect grids)
            correct_pixels = np.sum(predicted_grid == target_grid)
            total_pixels = target_grid.size
            pixel_accuracy = (correct_pixels / total_pixels) * 2.0  # Scale up pixel reward
            
            # Perfect grid bonus
            if correct_pixels == total_pixels:
                pixel_accuracy += 0.5  # Bonus for perfect match
        else:
            size_accuracy = -0.1  # Lighter penalty for wrong shape
    else:
        # CURRICULUM LEARNING: Start with lighter format penalty
        format_penalty_strength = min(0.1 + (current_step * 0.01), 0.5)  # 0.1 → 0.5 over 40 steps (gentler)
        format_reward = -format_penalty_strength
    
    # Calculate total reward
    total_reward = format_reward + size_accuracy + pixel_accuracy - final_channel_penalty
    
    return {
        'format_reward': format_reward,
        'size_accuracy': size_accuracy,
        'pixel_accuracy': pixel_accuracy,
        'final_channel_penalty': final_channel_penalty,
        'total_reward': total_reward,
        'has_grid': predicted_grid is not None,
        'grid_shape': predicted_grid.shape if predicted_grid is not None else None
    }

def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Main reward function for GRPO training
    Using 4-component reward system
    """
    global current_training_step, all_targets
    
    rewards = []
    step_info = []
    
    logger.info(f"Computing rewards for {len(completions)} completions at step {current_training_step}")
    
    for i, completion in enumerate(completions):
        # Get target grid for this completion
        target_idx = i % len(all_targets)
        target_grid = all_targets[target_idx]
        
        # Compute 4-component reward with curriculum learning
        reward_components = compute_four_component_reward(completion, target_grid, use_final_channel=True, current_step=current_training_step)
        
        total_reward = reward_components['total_reward']
        rewards.append(total_reward)
        
        step_info.append({
            'completion_idx': i,
            'target_idx': target_idx,
            'target_shape': target_grid.shape,
            **reward_components
        })
        
        # Log detailed info for first few completions
        if i < 3:
            logger.info(f"Completion {i}: format={reward_components['format_reward']:.3f}, "
                       f"size={reward_components['size_accuracy']:.3f}, "
                       f"pixel={reward_components['pixel_accuracy']:.3f}, "
                       f"final_penalty={reward_components['final_channel_penalty']:.3f}, "
                       f"total={total_reward:.3f}")
    
    # Log summary statistics
    if rewards:
        avg_reward = np.mean(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)
        std_reward = np.std(rewards)
        
        grids_found = sum(1 for info in step_info if info['has_grid'])
        perfect_matches = sum(1 for r in rewards if r >= 0.9)
        
        logger.info(f"Step {current_training_step} reward summary: "
                   f"avg={avg_reward:.4f}±{std_reward:.4f}, "
                   f"range=[{min_reward:.3f}, {max_reward:.3f}], "
                   f"grids_found={grids_found}/{len(completions)}, "
                   f"perfect={perfect_matches}")
        
        # WandB logging
        if wandb.run:
            wandb.log({
                "reward_mean": avg_reward,
                "reward_max": max_reward,
                "reward_min": min_reward,
                "reward_std": std_reward,
                "grids_extracted": grids_found,
                "perfect_solutions": perfect_matches,
                "step": current_training_step
            })
        
        # Save detailed info
        try:
            save_step_info(current_training_step, step_info, completions[:3])  # Save first 3 completions
        except Exception as e:
            logger.warning(f"Failed to save step info: {e}")
    
    current_training_step += 1
    return rewards

def save_step_info(step: int, step_info: List[Dict], sample_completions: List[str]):
    """Save detailed step information for analysis"""
    try:
        RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
        
        step_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "num_completions": len(step_info),
            "reward_components": step_info,
            "sample_completions": sample_completions[:3]  # Save first 3 for inspection
        }
        
        filename = RESPONSE_DIR / f"step_{step:05d}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(step_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved step {step} info to {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save step info: {e}")

def main_hf_trl_dapo():
    """Main DAPO training with HuggingFace TRL + QLoRA"""
    
    # Fix for vLLM/DeepSpeed blocking issues  
    os.environ["VLLM_USE_V1"] = "0"  # Disable vLLM V1 engine
    os.environ["NCCL_CUMEM_ENABLE"] = "0"  # Disable NCCL cuMem allocator
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # Use GPU 5
    
    # Force PyTorch to use GPU 2
    # Respect external CUDA_VISIBLE_DEVICES (e.g., run_gpt_oss_dapo.sh sets GPU 5)
    # Do not override device selection here.
    
    logger.info("🚀 Starting GPT-OSS ARC DAPO Training (HuggingFace TRL + QLoRA)")
    logger.info("="*70)
    
    try:
        # Setup WandB
        wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
        wandb.init(
            project="gpt-oss-arc-hf-trl-dapo",
            name=f"hf_trl_dapo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": "openai/gpt-oss-20b",
                "learning_rate": 1e-5,
                "batch_size": 1,
                "num_problems": 3,
                "reward_components": 4
            }
        )
        
        # Load model and tokenizer with QLoRA
        logger.info("Loading model and tokenizer...")
        
        # Skip quantization for compatibility
        # bnb_config = BitsAndBytesConfig(...) - Disabled due to version conflicts
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "openai/gpt-oss-20b", 
            trust_remote_code=True,
            padding_side="left",
        )
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model; prefer Flash Attention 2 when available, else gracefully fallback
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "openai/gpt-oss-20b",
                device_map={"": 0},  # maps to GPU 5 via CUDA_VISIBLE_DEVICES
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                "openai/gpt-oss-20b",
                device_map={"": 0},
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        
        # Add LoRA adapters
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"],  # Include MoE layers
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        logger.info("✅ Model and tokenizer loaded successfully")
        
        # Prepare dataset - Start with 3 problems
        problems = train_problems[:5]
        logger.info(f"📊 Loaded {len(problems)} ARC problems")
        
        # Create prompts and store targets globally
        prompts = []
        global all_targets
        all_targets = []
        
        for problem in problems:
            # Get prompt string directly with proper channel format
            prompt = create_arc_prompt_harmony(problem)
            prompts.append(prompt)
            all_targets.append(problem.test_pairs[0].y)
        
        # Create dataset
        dataset_dict = {
            "prompt": prompts,
            "problem_id": [p.uid for p in problems]
        }
        
        train_dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"📊 Created dataset with {len(train_dataset)} examples")
        
        # 🔥 paste.txt style stable DeepSpeed configuration from long_with_logit_reward_dapo.py
        deepspeed_config = {
            "train_batch_size": 4,  # batch_size * gradient_accumulation_steps
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu", 
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 5e7,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e4,
                "stage3_max_live_parameters": 1e8,
                "stage3_max_reuse_distance": 1e8,
                "stage3_gather_16bit_weights_on_model_save": True,
                "round_robin_gradients": True
            },
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
            "steps_per_print": 10,
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "activation_checkpointing": {
                "partition_activations": False,  # Disabled for stability
                "cpu_checkpointing": False,  # Disabled for stability
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            },
            "wall_clock_breakdown": False,
            "memory_breakdown": False
        }
        
        # GRPO Config with DAPO settings - Disable DeepSpeed for now
        grpo_config = GRPOConfig(
            output_dir=str(DATA_DIR / "checkpoints_hf_trl_dapo"),
            learning_rate=1e-5,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,  # Reduced for faster iteration
            warmup_steps=5,
            logging_steps=1,
            save_steps=10,
            bf16=True,
            fp16=False,
            optim="adamw_torch",
            report_to="wandb",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            # deepspeed=deepspeed_config,  # Disabled for simpler debugging
            
            # GRPO/DAPO specific parameters
            num_iterations=1,
            epsilon=0.2,
            epsilon_high=0.28,  # DAPO clip-higher
            beta=0.0,  # Remove KL term
            loss_type="bnpo",  # Use BNPO for DAPO token-level loss
            max_prompt_length=512,  # Reduced for stability
            max_completion_length=2048,  # Reduced for faster generation
            num_generations=2,  # Minimum required for GRPO
            generation_batch_size=2,  # Match num_generations
            max_steps=100,  # Increase training steps for better learning
        )
        
        # Create trainer
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_function,  # Use our 4-component reward function
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        
        # Build robust stopping based on Harmony stop tokens (string sequences → token id sequences)
        stop_id_sequences: List[List[int]] = []
        for tok in harmony_stop_tokens:
            try:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                if isinstance(ids, list) and len(ids) > 0:
                    stop_id_sequences.append(ids)
            except Exception:
                continue

        # Set generation kwargs after trainer initialization - HARMONY-ALIGNED SETTINGS
        trainer.generation_kwargs = {
            "max_new_tokens": 2048,  # Reasonable length for ARC responses
            "temperature": 0.5,      # Focused generation
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            # Keep standard EOS; rely on stopping_criteria for multi-token stops
            "eos_token_id": tokenizer.eos_token_id,
            # Softer repetition controls to avoid breaking structural tokens
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            # Robust stopping on Harmony sequences
            "stopping_criteria": StoppingCriteriaList([StopOnSequences(stop_id_sequences)]) if stop_id_sequences else None,
        }
        
        logger.info("🚀 Starting HuggingFace TRL DAPO training...")
        logger.info("📝 Using corrected Harmony format - no examples in input prompts")
        logger.info(f"   Loss type: {grpo_config.loss_type}")
        logger.info(f"   Epsilon: {grpo_config.epsilon}")
        logger.info(f"   Epsilon high: {grpo_config.epsilon_high}")
        logger.info(f"   Repetition penalty: {grpo_config.repetition_penalty}")
        logger.info(f"   Reward components: 4 (format + size + pixel + final_channel)")
        
        # Train
        trainer.train()
        
        logger.info("✅ Training completed!")
        
        # Save model
        model.save_pretrained(DATA_DIR / "final_model_hf_trl_dapo")
        tokenizer.save_pretrained(DATA_DIR / "final_model_hf_trl_dapo")
        logger.info(f"💾 Model saved to {DATA_DIR / 'final_model_hf_trl_dapo'}")
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        wandb.finish()

if __name__ == "__main__":
    main_hf_trl_dapo()
