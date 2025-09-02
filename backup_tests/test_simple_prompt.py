#!/usr/bin/env python3
"""
Simple test to check final channel reachability
"""

import os
import torch
import numpy as np
from arc import train_problems

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format"""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

def test_single_prompt():
    """Test a single prompt to see if it reaches final channel"""
    from unsloth import FastLanguageModel
    
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-BF16",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto"
    )
    
    # Test problem
    problem = train_problems[0]
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(f"Example {i}: Input:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}\n")
    examples_text = '\n'.join(examples)
    test_input = problem.test_pairs[0].x
    
    # Test final-only first (known to work)
    prompt = f"""<|start|>user<|message|>Solve this pattern:
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|channel|>final<|message|>"""
    
    print("Testing final-only prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated = response[len(prompt):]
    
    print(f"\nGenerated length: {len(generated)}")
    print(f"First 500 chars:\n{generated[:500]}")
    print(f"\nLast 500 chars:\n{generated[-500:]}")
    
    # Check for grid
    if "0" in generated or "1" in generated or "2" in generated:
        print("\n✅ Contains grid numbers")
    else:
        print("\n❌ No grid numbers found")

if __name__ == "__main__":
    test_single_prompt()