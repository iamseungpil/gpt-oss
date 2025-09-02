#!/usr/bin/env python3
"""
Test if GPT-OSS naturally generates final channel for ARC tasks
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arc import train_problems
import numpy as np

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

def grid_to_string(grid: np.ndarray) -> str:
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

# Prepare ARC problem
problem = train_problems[0]
examples_text = []
for i, train_pair in enumerate(problem.train_pairs):
    examples_text.append(
        f"Example {i+1}:\nInput:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}"
    )

# Create prompt with explicit channel instruction
prompt = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Reasoning: high<|end|><|start|>developer<|message|>You solve ARC puzzles. Use the analysis channel for reasoning and the final channel for your answer.
Example format:
<|channel|>analysis<|message|>
[Your reasoning here]
<|channel|>final<|message|>
[Final grid answer]<|end|><|start|>user<|message|>Solve this ARC puzzle:

{chr(10).join(examples_text)}

Test Input:
{grid_to_string(problem.test_pairs[0].x)}

Analyze the pattern and provide the output grid.<|end|><|start|>assistant"""

# Generate
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

print("\nGenerating response...")
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=2000,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)

# Check for channels
has_analysis = "<|channel|>analysis" in response
has_final = "<|channel|>final" in response

print("\n" + "="*50)
print("RESULTS:")
print(f"✓ Has analysis channel: {has_analysis}")
print(f"✓ Has final channel: {has_final}")
print("="*50)

if has_final:
    print("\n🎉 SUCCESS! Model naturally switches to final channel!")
else:
    print("\n⚠️ Model did not switch to final channel")

# Show first 500 chars of response
print("\nFirst 500 chars of response:")
print(response[:500])

# Save full response
with open("/tmp/arc_channel_test.txt", "w") as f:
    f.write(f"Has analysis channel: {has_analysis}\n")
    f.write(f"Has final channel: {has_final}\n")
    f.write("\n" + "="*50 + "\n")
    f.write(response)

print(f"\nFull response saved to /tmp/arc_channel_test.txt")