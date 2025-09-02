#!/usr/bin/env python3
"""
Test GPT-OSS inference with proper harmony format on GPU 5
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arc import train_problems
import numpy as np
from datetime import datetime

def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format"""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

print("🚀 Loading GPT-OSS model on GPU 5...")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "openai/gpt-oss-20b",
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("✅ Model loaded successfully")

# Get first problem
problem = train_problems[0]

# Build examples
examples = []
for i, train_pair in enumerate(problem.train_pairs, 1):
    examples.append(f"Example {i}:\nInput:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}")

test_input = problem.test_pairs[0].x
examples_text = '\n\n'.join(examples)

# Build prompt EXACTLY as in training
prompt = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high<|end|><|start|>developer<|message|># ARC Puzzle Solver

You solve ARC puzzles by:
1. Analyzing patterns in training examples 
2. Identifying transformation rules
3. Applying the rule to the test input

IMPORTANT: Use the analysis channel for reasoning, then switch to the final channel for your grid solution.

Format your response like this:
<|channel|>analysis<|message|>
[Your pattern analysis and reasoning here]
<|channel|>final<|message|>
[Your final grid answer here, just the numbers]<|end|><|start|>user<|message|>Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

Analyze the pattern, then provide the solution grid.<|end|><|start|>assistant"""

print("\n" + "="*70)
print("TESTING INFERENCE WITH PROPER PROMPT")
print("="*70)

# Tokenize
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
print(f"Prompt tokens: {input_ids.shape[1]}")

# Generate
print("\nGenerating response...")
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=2000,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[200002],  # <|return|> token
    )

# Decode response
response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)

print("\n" + "="*70)
print("RESPONSE ANALYSIS:")
print("="*70)

# Check for channels
has_analysis_channel = "<|channel|>analysis" in response
has_final_channel = "<|channel|>final" in response
has_message_token = "<|message|>" in response
has_return_token = "<|return|>" in response

print(f"✓ Has <|channel|>analysis: {has_analysis_channel}")
print(f"✓ Has <|channel|>final: {has_final_channel}")
print(f"✓ Has <|message|>: {has_message_token}")
print(f"✓ Has <|return|>: {has_return_token}")

# Check response structure
if response.startswith("<|channel|>"):
    print("\n✅ Response starts with channel token!")
else:
    print(f"\n⚠️ Response starts with: {response[:50]}")

print("\n" + "="*70)
print("FIRST 1000 CHARACTERS OF RESPONSE:")
print("="*70)
print(response[:1000])

# Save full response
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"/tmp/gpt_oss_inference_{timestamp}.txt"
with open(output_file, 'w') as f:
    f.write("PROMPT:\n")
    f.write("="*70 + "\n")
    f.write(prompt)
    f.write("\n\n" + "="*70 + "\n")
    f.write("RESPONSE:\n")
    f.write("="*70 + "\n")
    f.write(response)
    f.write("\n\n" + "="*70 + "\n")
    f.write("ANALYSIS:\n")
    f.write(f"Has analysis channel: {has_analysis_channel}\n")
    f.write(f"Has final channel: {has_final_channel}\n")
    f.write(f"Has message token: {has_message_token}\n")
    f.write(f"Has return token: {has_return_token}\n")

print(f"\n💾 Full response saved to: {output_file}")

# If channel switching works
if has_analysis_channel and has_final_channel:
    print("\n🎉 SUCCESS! Model uses both analysis and final channels!")
    
    # Try to extract final answer
    final_start = response.find("<|channel|>final<|message|>")
    if final_start != -1:
        final_content = response[final_start + len("<|channel|>final<|message|>"):]
        end_pos = final_content.find("<|")
        if end_pos != -1:
            final_content = final_content[:end_pos]
        print("\nFINAL ANSWER EXTRACTED:")
        print(final_content[:500])
elif has_analysis_channel:
    print("\n⚠️ Model uses analysis channel but doesn't switch to final")
else:
    print("\n❌ Model doesn't use proper channel format")