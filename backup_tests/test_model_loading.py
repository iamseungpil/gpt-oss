#!/usr/bin/env python3
"""
Test GPT-OSS model loading and generation
"""

import os
import torch
import numpy as np
from pathlib import Path

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.disable = True

def test_model_loading():
    """Test basic model loading and generation"""
    from unsloth import FastLanguageModel
    
    print("🔍 Testing GPT-OSS 20B model loading...")
    
    # Load model with same config
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-BF16",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map={"": 0},
        use_cache=False,
        trust_remote_code=True
    )
    
    print("✅ Model loaded successfully")
    print(f"Model config: {model.config}")
    print(f"Vocab size: {model.config.vocab_size}")
    print(f"Hidden size: {model.config.hidden_size}")
    
    # Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42
    )
    
    print("✅ LoRA added successfully")
    
    # Test simple generation
    test_prompt = "<|start|>system<|message|>You are a helpful assistant.<|end|>\n<|start|>user<|message|>Hello<|end|>\n<|start|>assistant<|message|>"
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    print(f"Input shape: {inputs.input_ids.shape}")
    
    # Generate with smaller max_new_tokens
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Response: {response}")
    
    return model, tokenizer

def test_grpo_setup():
    """Test GRPO setup with the model"""
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    
    print("\n🔍 Testing GRPO setup...")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-BF16",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map={"": 0},
        use_cache=False,
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
        use_gradient_checkpointing="unsloth",
        random_state=42
    )
    
    # Create minimal dataset
    data = [{"prompt": "Hello", "target": "Hi"}]
    dataset = Dataset.from_list(data)
    
    # Configure training with minimal settings
    training_args = GRPOConfig(
        output_dir="/tmp/test_grpo",
        max_steps=1,
        per_device_train_batch_size=1,
        torch_compile=False,
        bf16=True,
        logging_steps=1,
    )
    
    # Simple reward function
    def simple_reward(prompts, completions, trainer_state, **kwargs):
        return [0.0] * len(completions)
    
    # Try to create trainer
    try:
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset,
            reward_funcs=[simple_reward],
        )
        print("✅ GRPO trainer created successfully")
        
        # Try one training step
        trainer.train()
        print("✅ Training step completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*50)
    print("GPT-OSS 20B Test Script")
    print("="*50)
    
    # Test 1: Basic loading
    try:
        model, tokenizer = test_model_loading()
        print("\n✅ Model loading test passed")
    except Exception as e:
        print(f"\n❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    # Test 2: GRPO setup
    try:
        test_grpo_setup()
        print("\n✅ GRPO setup test passed")
    except Exception as e:
        print(f"\n❌ GRPO setup test failed: {e}")