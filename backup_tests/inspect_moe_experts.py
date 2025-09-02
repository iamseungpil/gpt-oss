#!/usr/bin/env python3
"""
GPT-OSS MoE Expert Structure Inspector
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from transformers import AutoModelForCausalLM
import torch

def inspect_moe_experts():
    print("🔍 Loading GPT-OSS to inspect MoE expert structure...")
    
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/gpt-oss-20b-BF16",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("\n🧠 MoE Expert Internal Structure:")
    
    # Check first layer's MLP structure
    first_layer = model.model.layers[0]
    mlp = first_layer.mlp
    
    print(f"MLP Type: {type(mlp)}")
    print(f"Router Type: {type(mlp.router)}")
    print(f"Experts Type: {type(mlp.experts)}")
    
    print(f"\nRouter modules:")
    for name, module in mlp.router.named_modules():
        if name:  # Skip the root module
            print(f"  router.{name}: {type(module).__name__}")
    
    print(f"\nExperts modules:")
    for name, module in mlp.experts.named_modules():
        if name:  # Skip the root module  
            print(f"  experts.{name}: {type(module).__name__}")
            if hasattr(module, 'weight'):
                print(f"    Shape: {module.weight.shape}")
    
    # Check if experts have gate_proj, up_proj, down_proj
    print(f"\n🎯 Expert Layer Names (first 5 experts):")
    expert_modules = []
    for name, module in mlp.named_modules():
        if 'experts' in name and ('gate' in name or 'up' in name or 'down' in name or 'proj' in name):
            expert_modules.append(name)
    
    for module_name in expert_modules[:10]:  # Show first 10
        print(f"  {module_name}")
    
    print(f"\n💡 Recommended LoRA Target Modules for GPT-OSS MoE:")
    attention_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    expert_targets = []
    
    # Extract unique expert module names
    for module_name in expert_modules:
        parts = module_name.split('.')
        if len(parts) >= 3:  # experts.0.gate_proj etc
            layer_name = parts[-1]  # gate_proj, up_proj, down_proj
            if layer_name not in expert_targets:
                expert_targets.append(layer_name)
    
    all_targets = attention_targets + expert_targets
    print(f"  Attention: {attention_targets}")
    print(f"  MoE Expert: {expert_targets}")  
    print(f"  Complete: {all_targets}")

if __name__ == "__main__":
    inspect_moe_experts()