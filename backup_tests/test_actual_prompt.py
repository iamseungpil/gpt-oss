#!/usr/bin/env python3
"""Test the actual prompt being used in training"""

from arc import train_problems
import numpy as np

def grid_to_string(grid: np.ndarray) -> str:
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

# Get first problem
problem = train_problems[0]

# Build examples section
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

print("PROMPT LENGTH:", len(prompt))
print("\n" + "="*50)
print("FIRST 1500 CHARS:")
print("="*50)
print(prompt[:1500])

print("\n" + "="*50)
print("CHECKING CONTENT:")
print("="*50)
print(f"Has '<|channel|>analysis': {'<|channel|>analysis' in prompt}")
print(f"Has '<|channel|>final': {'<|channel|>final' in prompt}")
print(f"Has 'analysis channel': {'analysis channel' in prompt}")
print(f"Has 'ARC Puzzle Solver': {'ARC Puzzle Solver' in prompt}")

# Check developer message
dev_start = prompt.find("<|start|>developer")
dev_end = prompt.find("<|end|>", dev_start)
if dev_start != -1 and dev_end != -1:
    dev_content = prompt[dev_start:dev_end+7]
    print(f"\nDeveloper message length: {len(dev_content)} chars")
    print("Developer message contains channel examples:", "<|channel|>" in dev_content)