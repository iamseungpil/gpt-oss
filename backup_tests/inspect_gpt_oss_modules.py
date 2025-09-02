#!/usr/bin/env python3
"""
GPT-OSS Model Architecture Inspector
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from transformers import AutoModelForCausalLM
import torch

def inspect_gpt_oss_modules():
    print("🔍 Loading GPT-OSS 20B to inspect modules...")
    
    # Load model config only to check architecture
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/gpt-oss-20b-BF16",
        device_map="cpu",  # CPU only for inspection
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("\n📋 GPT-OSS Model Architecture:")
    print(f"Model type: {type(model)}")
    print(f"Config: {model.config}")
    
    print("\n🧱 Layer Structure:")
    for name, module in model.named_modules():
        if any(keyword in name for keyword in ['proj', 'gate', 'expert', 'router', 'mlp', 'attention']):
            print(f"  {name}: {type(module).__name__}")
    
    print("\n🎯 Recommended LoRA Target Modules:")
    target_modules = []
    for name, module in model.named_modules():
        if 'proj' in name and any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
            if 'expert' not in name:  # Avoid expert-specific modules for now
                target_modules.append(name.split('.')[-1])
    
    unique_targets = list(set(target_modules))
    print(f"Suggested: {unique_targets}")
    
    # Check embedding and lm_head dimensions
    print(f"\n📊 Model Dimensions:")
    if hasattr(model, 'get_input_embeddings'):
        embed = model.get_input_embeddings()
        print(f"Input embeddings: {embed.weight.shape}")
    
    if hasattr(model, 'get_output_embeddings'):
        output = model.get_output_embeddings()  
        if output is not None:
            print(f"Output embeddings: {output.weight.shape}")
    
    print(f"Vocab size: {model.config.vocab_size}")
    print(f"Hidden size: {model.config.hidden_size}")

if __name__ == "__main__":
    inspect_gpt_oss_modules()