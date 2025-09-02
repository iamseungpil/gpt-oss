#!/usr/bin/env python3
"""
Test the final corrected prompt format
"""

from arc import train_problems
import numpy as np

def grid_to_string(grid: np.ndarray) -> str:
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

# Get first problem
problem = train_problems[0]

# Build examples
examples = []
for i, train_pair in enumerate(problem.train_pairs[:2], 1):  # Just 2 examples for testing
    examples.append(f"Example {i}:\nInput:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}")

test_input = problem.test_pairs[0].x
examples_text = '\n\n'.join(examples)

# Build the CORRECTED prompt
prompt = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># ARC Puzzle Solver

You are an expert at solving Abstract Reasoning Corpus (ARC) puzzles.

## Instructions
You MUST use channels to structure your response:
1. Use <|channel|>analysis<|message|> for identifying patterns and reasoning
2. Use <|channel|>final<|message|> for providing the final solution grid

## Example Response Format
<|channel|>analysis<|message|>
Looking at the examples, I can see that...
[detailed pattern analysis]
<|channel|>final<|message|>
[solution grid with numbers only]<|end|><|start|>user<|message|>Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

Analyze the pattern, then provide the solution grid.<|end|><|start|>assistant<|channel|>"""

print("CORRECTED PROMPT FORMAT TEST")
print("="*70)
print(f"Prompt length: {len(prompt)} chars")
print(f"\nPrompt ends with: '{prompt[-100:]}'")
print("\n" + "="*70)
print("KEY FEATURES:")
print("="*70)
print("✓ System message includes: 'Reasoning: high'")
print("✓ System message includes: 'Valid channels: analysis, commentary, final'")
print("✓ Developer message is NOT empty")
print("✓ Developer message includes channel examples")
print("✓ Prompt ends with: '<|start|>assistant<|channel|>'")
print("\n" + "="*70)
print("EXPECTED MODEL BEHAVIOR:")
print("="*70)
print("1. Model should continue with: 'analysis<|message|>'")
print("2. Then provide pattern analysis")
print("3. Then switch to: '<|channel|>final<|message|>'")
print("4. Then provide the solution grid")

# Save the prompt for inspection
with open("/tmp/corrected_prompt.txt", "w") as f:
    f.write(prompt)

print("\n💾 Full prompt saved to /tmp/corrected_prompt.txt")
print(f"\nPrompt is ready for training/inference!")