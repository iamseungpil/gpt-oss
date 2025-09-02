#!/usr/bin/env python3
"""
Test base GPT-OSS 20B model (without fine-tuning) for channel generation
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arc import train_problems
import numpy as np

def grid_to_string(grid: np.ndarray) -> str:
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

print("🚀 Loading BASE GPT-OSS 20B model (no fine-tuning)...")

# Load BASE model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "openai/gpt-oss-20b",
    trust_remote_code=True,
)

# Load with 8-bit quantization to save memory
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

print("✅ BASE model loaded successfully")

# Test 1: Simple channel test
print("\n" + "="*70)
print("TEST 1: Simple Math with Channels")
print("="*70)

simple_prompt = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high<|end|><|start|>developer<|message|>Always use channels to structure your response:
<|channel|>analysis<|message|>
For your step-by-step reasoning
<|channel|>final<|message|>
For your final answer<|end|><|start|>user<|message|>What is 15 + 27? Show your work using channels.<|end|><|start|>assistant"""

input_ids = tokenizer(simple_prompt, return_tensors="pt").input_ids.to(model.device)

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[200002],  # <|return|>
    )

response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)

print("Response:")
print(response[:500])
print(f"\n✓ Has <|channel|>analysis: {'<|channel|>analysis' in response}")
print(f"✓ Has <|channel|>final: {'<|channel|>final' in response}")

# Test 2: ARC puzzle with channels
print("\n" + "="*70)
print("TEST 2: ARC Puzzle with Channels")
print("="*70)

problem = train_problems[0]
examples = []
for i, train_pair in enumerate(problem.train_pairs[:2], 1):  # Use only 2 examples for brevity
    examples.append(f"Example {i}:\nInput:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}")

test_input = problem.test_pairs[0].x
examples_text = '\n\n'.join(examples)

arc_prompt = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high<|end|><|start|>developer<|message|>You solve ARC puzzles step by step.

IMPORTANT: Structure your response using channels:
<|channel|>analysis<|message|>
Analyze the pattern here
<|channel|>final<|message|>
Provide the final grid here<|end|><|start|>user<|message|>Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

Use the analysis channel to explain the pattern, then the final channel for your answer.<|end|><|start|>assistant"""

input_ids = tokenizer(arc_prompt, return_tensors="pt").input_ids.to(model.device)
print(f"Prompt tokens: {input_ids.shape[1]}")

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=1000,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[200002],
    )

response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)

print("\nResponse (first 800 chars):")
print(response[:800])

print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)
print(f"✓ Has <|channel|>analysis: {'<|channel|>analysis' in response}")
print(f"✓ Has <|channel|>final: {'<|channel|>final' in response}")
print(f"✓ Has <|message|>: {'<|message|>' in response}")

if response.startswith("<|channel|>"):
    print("\n🎉 SUCCESS! Base model naturally uses channel tokens!")
    # Extract channel content
    if "<|channel|>analysis" in response:
        analysis_start = response.find("<|channel|>analysis<|message|>")
        if analysis_start != -1:
            analysis_content = response[analysis_start + len("<|channel|>analysis<|message|>"):]
            end_pos = analysis_content.find("<|channel|>")
            if end_pos == -1:
                end_pos = analysis_content.find("<|end|>")
            if end_pos != -1:
                analysis_content = analysis_content[:end_pos]
            print("\nANALYSIS CHANNEL CONTENT:")
            print(analysis_content[:300])
else:
    print(f"\n⚠️ Base model doesn't start with channel token")
    print(f"Response starts with: '{response[:50]}...'")

# Save full response
with open("/tmp/base_model_test.txt", "w") as f:
    f.write("BASE GPT-OSS 20B MODEL TEST\n")
    f.write("="*70 + "\n\n")
    f.write("TEST 1 - Simple Math:\n")
    f.write(response + "\n\n")
    f.write("="*70 + "\n")
    f.write("TEST 2 - ARC Puzzle:\n")
    f.write(response + "\n")

print("\n💾 Full results saved to /tmp/base_model_test.txt")