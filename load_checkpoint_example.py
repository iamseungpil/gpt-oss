#!/usr/bin/env python3
"""
GPT-OSS Checkpoint Loading Example
Shows how to load a trained checkpoint for inference
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def load_checkpoint_for_inference(checkpoint_path):
    """Load a trained checkpoint for inference."""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint path does not exist: {checkpoint_path}")
        return None, None
    
    print(f"🔄 Loading checkpoint from: {checkpoint_path}")
    
    # Load training info if available
    training_info_path = os.path.join(checkpoint_path, "training_info.json")
    if os.path.exists(training_info_path):
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
        print(f"📋 Training info:")
        for key, value in training_info.items():
            print(f"   {key}: {value}")
        print("=" * 50)
    
    # Load tokenizer
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        padding_side='left'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Tokenizer loaded")
    
    # Load model
    print("🔄 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",  # Automatically place on available GPUs
    )
    
    print(f"✅ Model loaded: {model.num_parameters():,} parameters")
    print(f"🧠 Model device: {next(model.parameters()).device}")
    
    return model, tokenizer

def test_inference(model, tokenizer, test_prompt="Hello, how are you?"):
    """Test inference with the loaded model."""
    
    print(f"🧪 Testing inference with prompt: '{test_prompt}'")
    
    # Create messages for chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": test_prompt}
    ]
    
    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="medium"
        )
    except:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    
    print(f"🤖 Model response:")
    print(response)
    print("=" * 50)
    
    return response

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and test GPT-OSS checkpoint")
    parser.add_argument("checkpoint_path", help="Path to checkpoint directory")
    parser.add_argument("--test_prompt", type=str, default="Solve this ARC puzzle pattern.", 
                       help="Test prompt for inference")
    
    args = parser.parse_args()
    
    # Load checkpoint
    model, tokenizer = load_checkpoint_for_inference(args.checkpoint_path)
    
    if model is None or tokenizer is None:
        print("❌ Failed to load checkpoint")
        sys.exit(1)
    
    # Test inference
    test_inference(model, tokenizer, args.test_prompt)
    
    print("✅ Checkpoint loading and testing completed!")

if __name__ == "__main__":
    main()