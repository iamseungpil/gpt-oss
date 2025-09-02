#!/usr/bin/env python3
"""
GPT-OSS GRPO Training for ARC-AGI with Harmony Format
GRPO (Group Relative Policy Optimization) for teaching proper channel switching
"""

import os
import json
import random
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import numpy as np
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="openai/gpt-oss-20b")
    use_flash_attention: bool = field(default=True)
    trust_remote_code: bool = field(default=True)


@dataclass
class DataArguments:
    arc_data_path: str = field(
        default="/home/ubuntu/barc_feedback/SOAR-main/arc-prize-2025/arc-agi_training_challenges.json"
    )
    max_train_samples: int = field(default=100)
    val_split: float = field(default=0.1)


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default="/data/gpt_oss_final/grpo_harmony")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    gradient_checkpointing: bool = field(default=True)
    learning_rate: float = field(default=1e-5)
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=100)
    save_total_limit: int = field(default=3)
    push_to_hub: bool = field(default=False)
    report_to: List[str] = field(default_factory=lambda: ["none"])
    remove_unused_columns: bool = field(default=False)
    label_names: List[str] = field(default_factory=lambda: ["labels"])
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    optim: str = field(default="paged_adamw_8bit")
    lr_scheduler_type: str = field(default="cosine")
    deepspeed: Optional[str] = field(default=None)


@dataclass
class LoraArguments:
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.1)
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_bias: str = field(default="none")


class ARCDataProcessor:
    """Process ARC-AGI data for training"""
    
    def __init__(self, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_arc_data(self, path: str) -> Dict:
        """Load ARC training data"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def format_arc_problem(self, problem: Dict) -> str:
        """Format ARC problem with harmony instructions"""
        train_examples = []
        for i, example in enumerate(problem['train'], 1):
            train_examples.append(f"Example {i}:")
            train_examples.append(f"Input: {json.dumps(example['input'])}")
            train_examples.append(f"Output: {json.dumps(example['output'])}")
        
        test_input = json.dumps(problem['test'][0]['input'])
        
        user_content = f"""Solve this ARC (Abstraction and Reasoning Corpus) puzzle.

Training Examples:
{chr(10).join(train_examples)}

Test Input:
{test_input}

Analyze the pattern and provide the test output as a 2D grid."""

        # Harmony format messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert at solving abstract reasoning puzzles.\nReasoning: medium"
            },
            {
                "role": "developer",
                "content": """# CRITICAL INSTRUCTIONS - MUST FOLLOW

You MUST generate your response using EXACTLY TWO channels in this order:

1. FIRST: Use <|channel|>analysis<|message|> for your reasoning
   - Analyze the training examples
   - Identify the transformation pattern
   - Explain your logic step by step
   
2. SECOND: Use <|channel|>final<|message|> for the solution
   - Output ONLY the grid as [[row1], [row2], ...]
   - Numbers only, no explanation

IMPORTANT: You MUST include BOTH channels. Do NOT stop after analysis.
After your analysis, you MUST switch to final channel and output the grid.

Example structure:
<|channel|>analysis<|message|>[detailed reasoning]<|end|><|start|>assistant<|channel|>final<|message|>[[grid]]<|return|>"""
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return text
    
    def prepare_dataset(self, data_path: str, max_samples: int = 100) -> Dataset:
        """Prepare dataset for GRPO training"""
        arc_data = self.load_arc_data(data_path)
        
        # Sample problems
        problem_ids = list(arc_data.keys())[:max_samples]
        
        dataset_dict = {
            "prompt": [],
            "problem_id": []
        }
        
        for pid in problem_ids:
            problem = arc_data[pid]
            prompt = self.format_arc_problem(problem)
            
            dataset_dict["prompt"].append(prompt)
            dataset_dict["problem_id"].append(pid)
        
        return Dataset.from_dict(dataset_dict)


class HarmonyRewardModel:
    """Compute rewards for harmony format compliance and correctness"""
    
    def __init__(self):
        self.channel_bonus = 0.5  # Reward for using correct channels
        self.grid_bonus = 0.3     # Reward for valid grid format
        self.both_channels_bonus = 0.8  # Bonus for using both channels
        
    def extract_grid(self, text: str) -> Optional[List[List[int]]]:
        """Extract grid from response"""
        try:
            # Look for grid pattern [[...]]
            grid_match = re.search(r'\[\[[\d,\s\[\]]+\]\]', text)
            if grid_match:
                grid_str = grid_match.group()
                grid = eval(grid_str)  # Safe as we've validated format
                if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                    return grid
        except:
            pass
        return None
    
    def compute_reward(self, response: str, target_grid: Optional[List[List[int]]] = None) -> float:
        """
        Compute reward based on:
        1. Using analysis channel
        2. Using final channel  
        3. Proper channel switching
        4. Valid grid output
        5. Grid correctness (if target provided)
        """
        reward = 0.0
        
        # Check for analysis channel
        has_analysis = "<|channel|>analysis" in response
        if has_analysis:
            reward += 0.3
            
        # Check for final channel
        has_final = "<|channel|>final" in response
        if has_final:
            reward += 0.3
            
        # Bonus for using both channels
        if has_analysis and has_final:
            reward += self.both_channels_bonus
            
            # Check if final comes after analysis
            analysis_pos = response.find("<|channel|>analysis")
            final_pos = response.find("<|channel|>final")
            if final_pos > analysis_pos:
                reward += 0.2
        
        # Check for valid grid in final channel
        if has_final:
            final_content = response.split("<|channel|>final<|message|>")[-1]
            grid = self.extract_grid(final_content)
            if grid:
                reward += self.grid_bonus
                
                # If target provided, check correctness
                if target_grid:
                    try:
                        # Check shape match
                        if len(grid) == len(target_grid) and len(grid[0]) == len(target_grid[0]):
                            reward += 0.2
                            
                            # Pixel accuracy
                            correct = sum(
                                1 for i in range(len(grid))
                                for j in range(len(grid[0]))
                                if grid[i][j] == target_grid[i][j]
                            )
                            total = len(grid) * len(grid[0])
                            accuracy = correct / total
                            reward += accuracy * 0.5
                    except:
                        pass
        
        # Penalty for broken/repetitive output
        if response.count(response[:20]) > 10:  # Repetition detection
            reward -= 0.5
            
        return reward


def setup_model_and_tokenizer(model_args: ModelArguments):
    """Setup model and tokenizer with proper configurations"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True
    )
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (GPT-OSS already has MXFP4 quantization, no need for BitsAndBytes)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention else "eager"
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    return model, tokenizer


def main():
    """Main training function"""
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    os.environ["WANDB_DISABLED"] = "true"
    
    # Setup logging
    logger.info("Starting GPT-OSS GRPO training for harmony format")
    logger.info(f"Output directory: {training_args.output_dir}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # Setup LoRA
    logger.info("Setting up LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.target_modules,
        bias=lora_args.lora_bias,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    data_processor = ARCDataProcessor(tokenizer)
    train_dataset = data_processor.prepare_dataset(
        data_args.arc_data_path,
        data_args.max_train_samples
    )
    
    # Initialize reward model
    reward_model = HarmonyRewardModel()
    
    # GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        num_train_epochs=training_args.num_train_epochs,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        warmup_steps=training_args.warmup_steps,
        optim=training_args.optim,
        bf16=training_args.bf16,
        gradient_checkpointing=training_args.gradient_checkpointing,
        seed=42,
    )
    
    # Custom reward function for GRPO
    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
        """Compute rewards for GRPO"""
        rewards = []
        for output in outputs:
            reward = reward_model.compute_reward(output)
            rewards.append(reward)
            
            # Log channel usage
            has_analysis = "<|channel|>analysis" in output
            has_final = "<|channel|>final" in output
            logger.info(f"Channels - Analysis: {has_analysis}, Final: {has_final}, Reward: {reward:.3f}")
            
        return rewards
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_fn=reward_fn,
        peft_config=peft_config,
    )
    
    # Add generation kwargs for proper harmony format
    trainer.generation_kwargs = {
        "max_new_tokens": 10000,  # Very important!
        "temperature": 0.5,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": [tokenizer.eos_token_id, 200002],  # Include <|return|> token
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 4,
    }
    
    # Start training
    logger.info("Starting GRPO training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()