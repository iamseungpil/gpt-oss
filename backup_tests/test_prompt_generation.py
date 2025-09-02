#!/usr/bin/env python3
"""
Test prompt generation to see what model actually receives
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

def test_different_prompts():
    """Test different prompt formats to see which one generates final channel"""
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
    examples_text = '\n'.join(examples[:1])  # Use only first example for speed
    test_input = problem.test_pairs[0].x
    
    prompts = {
        "current_complex": f"""<|start|>system<|message|>You are a pattern recognition model. Analyze the given examples and solve the test case.

Reasoning: high

# Valid channels: analysis, commentary, final
# You MUST provide your answer in the final channel as a grid of numbers.
# First analyze the pattern in the analysis channel, then provide the grid in the final channel.<|end|>
<|start|>user<|message|>Solve this pattern:
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|channel|>analysis<|message|>Let me analyze the pattern.<|end|>
<|start|>assistant<|channel|>final<|message|>""",
        
        "simple_final": f"""<|start|>user<|message|>Output the grid:
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|channel|>final<|message|>""",
        
        "no_prefilled": f"""<|start|>user<|message|>Solve this pattern:
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|message|>""",
        
        "explicit_final": f"""<|start|>user<|message|>Solve this pattern. Output ONLY in final channel:
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|channel|>final<|message|>Based on the pattern, the output is:
""",
    }
    
    for name, prompt in prompts.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        print(f"Prompt preview (last 200 chars):\n...{prompt[-200:]}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Shorter for testing
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated = response[len(prompt):]
        
        print(f"\nGenerated ({len(generated)} chars):")
        print(generated[:500])
        
        # Check for key markers
        print(f"\nAnalysis:")
        print(f"  Contains '<|channel|>final<|': {'YES' if '<|channel|>final<|' in generated else 'NO'}")
        print(f"  Contains '<|channel|>analysis<|': {'YES' if '<|channel|>analysis<|' in generated else 'NO'}")
        print(f"  Contains grid numbers: {'YES' if any(str(i) in generated[:200] for i in range(10)) else 'NO'}")
        print(f"  Starts with grid: {'YES' if generated.strip() and generated.strip()[0].isdigit() else 'NO'}")

if __name__ == "__main__":
    test_different_prompts()