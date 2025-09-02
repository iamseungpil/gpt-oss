#!/usr/bin/env python3
"""
Simple test to verify model can generate without blocking
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["VLLM_USE_V1"] = "0"
os.environ["NCCL_CUMEM_ENABLE"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

print("🔧 Loading model for simple generation test...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("barc0/Llama-3.1-ARC-Potpourri-Induction-8B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with minimal config
model = AutoModelForCausalLM.from_pretrained(
    "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# Add simple LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # Reduced from 16
    lora_alpha=16,  # Reduced from 32
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("✅ Model loaded successfully")

# Test simple generation
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("🔧 Testing generation...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"✅ Generation successful: {response}")