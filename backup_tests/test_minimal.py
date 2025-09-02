#!/usr/bin/env python3
"""
Minimal test of base GPT-OSS model channel generation
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)

print("Loading model with bfloat16...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Very simple test
prompt = """<|start|>system<|message|>You are ChatGPT.
Reasoning: high<|end|><|start|>user<|message|>Say hello using the analysis channel.<|end|><|start|>assistant"""

print("\nPrompt:", prompt[:100], "...")

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

print(f"Generating (input: {input_ids.shape[1]} tokens)...")
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=50,
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
print(f"Starts with channel: {response.startswith('<|channel|>')}")

# Save result
with open("/tmp/minimal_test.txt", "w") as f:
    f.write(f"Prompt:\n{prompt}\n\n")
    f.write(f"Response:\n{response}\n")
    f.write(f"\nHas channel: {'<|channel|>' in response}\n")

print("\nSaved to /tmp/minimal_test.txt")