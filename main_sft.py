#!/usr/bin/env python3
"""
GPT-OSS 20B ARC Training with SFT (Supervised Fine-Tuning)
============================================================

Using standard SFT instead of DAPO to avoid compatibility issues.
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
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["WANDB_API_KEY"] = "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.disable = True

# Core imports
import wandb
from arc import train_problems
from datasets import Dataset

# Configuration
DATA_DIR = Path("/data/gpt_oss_final")
LOG_FILE = DATA_DIR / "logs" / "sft.log"

# Setup logging
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
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
    """Convert numpy grid to string format"""
    return '\\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

def create_arc_prompt(problem) -> Tuple[str, str]:
    """Create ARC prompt and expected response"""
    
    # Build examples section
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(f"Example {i}: Input:\\n{grid_to_string(train_pair.x)}\\nOutput:\\n{grid_to_string(train_pair.y)}\\n")
    
    # Test section
    test_input = problem.test_pairs[0].x
    test_output = problem.test_pairs[0].y
    examples_text = '\\n'.join(examples)
    
    # Create prompt
    prompt = f"""<|start|>system<|message|>You are a pattern solver. Use high reasoning.
Reasoning: high
Output format: Provide the final answer as a grid of numbers.<|end|>
<|start|>user<|message|>Solve this pattern and output the grid:
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|message|>"""
    
    # Expected response (with final channel)
    response = f"""Let me analyze this pattern step by step.

Looking at the examples, I can see the transformation rule.

The output grid is:
```
{grid_to_string(test_output)}
```<|end|>"""
    
    return prompt, response

def create_sft_dataset(problems: List) -> Dataset:
    """Create dataset for SFT training"""
    
    data = []
    for problem in problems:
        prompt, response = create_arc_prompt(problem)
        
        # Combine for training
        text = prompt + response
        
        data.append({
            "text": text,
            "prompt": prompt,
            "response": response,
            "problem_id": problem.uid
        })
    
    return Dataset.from_list(data)

def main_sft():
    """Main SFT training"""
    
    logger.info("🚀 Starting GPT-OSS ARC SFT Training")
    logger.info("="*50)
    
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        
        # Setup WandB
        wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
        wandb.init(
            project="gpt-oss-arc-sft",
            name=f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        
        # Load model
        logger.info("Loading model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gpt-oss-20b-BF16",
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map={"": 0},
            trust_remote_code=True
        )
        
        # Add LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=False,
            random_state=42
        )
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✅ Model loaded | Trainable params: {trainable_params/1e6:.2f}M")
        
        # Prepare dataset
        problems = train_problems[:10]  # Use more problems for SFT
        logger.info(f"📊 Loaded {len(problems)} ARC problems")
        
        train_dataset = create_sft_dataset(problems)
        logger.info(f"📊 Created dataset with {len(train_dataset)} examples")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(DATA_DIR / "checkpoints_sft"),
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_steps=10,
            logging_steps=1,
            save_steps=50,
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            report_to="wandb",
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            packing=False,
        )
        
        logger.info("🚀 Starting SFT training...")
        
        # Train
        trainer.train()
        
        logger.info("✅ Training completed!")
        
        # Save model
        model.save_pretrained(DATA_DIR / "final_model_sft")
        tokenizer.save_pretrained(DATA_DIR / "final_model_sft")
        logger.info(f"💾 Model saved")
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main_sft()