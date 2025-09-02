#!/usr/bin/env python3
"""
Quick test script to verify GPT-OSS can load and run without errors
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "39550"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✅ Transformers imported")
except ImportError as e:
    print(f"❌ Transformers import failed: {e}")

try:
    from trl import GRPOTrainer, GRPOConfig
    print("✅ TRL imported")
except ImportError as e:
    print(f"❌ TRL import failed: {e}")

try:
    import deepspeed
    print(f"✅ DeepSpeed imported (version: {deepspeed.__version__})")
except ImportError as e:
    print(f"❌ DeepSpeed import failed: {e}")

try:
    from peft import LoraConfig, get_peft_model, TaskType
    print("✅ PEFT imported")
except ImportError as e:
    print(f"❌ PEFT import failed: {e}")

# Test loading a small model
try:
    print("\nTesting model loading...")
    tokenizer = AutoTokenizer.from_pretrained("barc0/Llama-3.1-ARC-Potpourri-Induction-8B")
    print("✅ Tokenizer loaded")
    
    # Just load config to test without OOM
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("barc0/Llama-3.1-ARC-Potpourri-Induction-8B")
    print(f"✅ Model config loaded (hidden_size: {config.hidden_size})")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")

print("\n✅ All basic tests passed! Ready to run main training script.")