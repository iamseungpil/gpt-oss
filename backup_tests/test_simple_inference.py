#!/usr/bin/env python3
"""
Simple test to check if model generates channel tokens
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)

# Test 1: Check if channel tokens exist
test_prompts = [
    "<|channel|>analysis<|message|>Test",
    "<|channel|>final<|message|>Test",
]

print("\n" + "="*50)
print("TOKEN TEST:")
print("="*50)
for prompt in test_prompts:
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)
    print(f"Prompt: {prompt}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    print(f"Match: {decoded == prompt}\n")

# Test 2: Simple generation test
simple_prompt = """<|start|>system<|message|>You are ChatGPT.
Reasoning: high<|end|><|start|>developer<|message|>Use channels:
<|channel|>analysis<|message|>
For analysis
<|channel|>final<|message|>
For final answer<|end|><|start|>user<|message|>What is 2+2? Use channels.<|end|><|start|>assistant"""

print("="*50)
print("SIMPLE PROMPT TEST:")
print("="*50)
print("Prompt:", simple_prompt[:200], "...")

# Load model with minimal memory
print("\nLoading model with 8-bit quantization...")
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("Model loaded!")

# Generate
input_ids = tokenizer(simple_prompt, return_tensors="pt").input_ids.to(model.device)

print("\nGenerating...")
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)

print("\n" + "="*50)
print("RESPONSE:")
print("="*50)
print(response)

print("\n" + "="*50)
print("ANALYSIS:")
print("="*50)
print(f"Has <|channel|>: {'<|channel|>' in response}")
print(f"Has analysis: {'analysis' in response}")
print(f"Has final: {'final' in response}")

if response.startswith("<|channel|>"):
    print("\n✅ SUCCESS! Response starts with channel token!")
else:
    print(f"\n⚠️ Response starts with: {response[:30]}")