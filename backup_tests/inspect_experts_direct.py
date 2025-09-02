#!/usr/bin/env python3
"""
Direct GPT-OSS Expert Inspection
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from transformers import AutoModelForCausalLM
import torch

def inspect_experts_direct():
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/gpt-oss-20b-BF16",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Get first layer for inspection
    layer = model.model.layers[0]
    mlp = layer.mlp
    
    print("🔍 Direct MLP Attributes:")
    print(f"MLP dir: {[attr for attr in dir(mlp) if not attr.startswith('_')]}")
    
    print(f"\n🎯 Router Attributes:")
    router = mlp.router
    print(f"Router dir: {[attr for attr in dir(router) if not attr.startswith('_')]}")
    
    print(f"\n🧠 Experts Attributes:")
    experts = mlp.experts
    print(f"Experts dir: {[attr for attr in dir(experts) if not attr.startswith('_')]}")
    
    # Try to access experts directly
    print(f"\n🔬 Experts Internal:")
    try:
        if hasattr(experts, 'experts'):
            print(f"experts.experts exists")
            first_expert = experts.experts[0] if hasattr(experts.experts, '__getitem__') else None
            if first_expert:
                print(f"First expert type: {type(first_expert)}")
                print(f"First expert dir: {[attr for attr in dir(first_expert) if not attr.startswith('_')]}")
        
        if hasattr(experts, 'w1'):
            print(f"experts.w1 shape: {experts.w1.shape}")
        if hasattr(experts, 'w2'):  
            print(f"experts.w2 shape: {experts.w2.shape}")
        if hasattr(experts, 'w3'):
            print(f"experts.w3 shape: {experts.w3.shape}")
            
    except Exception as e:
        print(f"Error accessing experts: {e}")
    
    # Check all module names containing 'expert' 
    print(f"\n📋 All Expert-Related Module Names:")
    count = 0
    for name, module in model.named_modules():
        if 'expert' in name.lower():
            print(f"  {name}: {type(module).__name__}")
            if hasattr(module, 'weight') and count < 5:
                print(f"    Weight shape: {module.weight.shape}")
            count += 1
            if count >= 10:  # Limit output
                print("  ... (truncated)")
                break

if __name__ == "__main__":
    inspect_experts_direct()