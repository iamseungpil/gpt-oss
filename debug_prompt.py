#!/usr/bin/env python3
"""
Debug the actual prompt being generated for training
"""

from arc import train_problems
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    SystemContent,
    DeveloperContent,
    TextContent
)
from transformers import AutoTokenizer
import numpy as np

def grid_to_string(grid: np.ndarray) -> str:
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

# Load tokenizer and harmony
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Get first problem
problem = train_problems[0]

# Build examples section
examples = []
for i, train_pair in enumerate(problem.train_pairs, 1):
    examples.append(f"Example {i}:\nInput:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}")

test_input = problem.test_pairs[0].x
examples_text = '\n\n'.join(examples)

user_content = f"""Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

Analyze the pattern, then provide the solution grid."""

# Build conversation EXACTLY as in training
convo = Conversation.from_messages([
    Message.from_role_and_content(
        Role.SYSTEM,
        SystemContent(
            text="You are ChatGPT, a large language model trained by OpenAI.",
            knowledge_cutoff="2024-06",
            reasoning="high"  # Note: using high reasoning
        )
    ),
    Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent(
            text="""# ARC Puzzle Solver

You solve ARC puzzles by:
1. Analyzing patterns in training examples 
2. Identifying transformation rules
3. Applying the rule to the test input

Use the analysis channel for reasoning, then the final channel for your grid solution.

Example format:
<|channel|>analysis<|message|>
Pattern identified: Fill empty cells with 1s, keep original values.
<|channel|>final<|message|>
1 2 3
4 5 6
7 8 9"""
        )
    ),
    Message.from_role_and_content(
        Role.USER,
        TextContent(text=user_content)
    ),
])

# Render for completion
token_ids = harmony_encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

# Decode back to text (as training does)
decoded_prompt = tokenizer.decode(token_ids, skip_special_tokens=False)

print("="*70)
print("DECODED PROMPT (first 2000 chars):")
print("="*70)
print(decoded_prompt[:2000])

print("\n" + "="*70)
print("CHECKING FOR CHANNEL INSTRUCTIONS:")
print("="*70)
print(f"Contains '<|channel|>analysis': {'<|channel|>analysis' in decoded_prompt}")
print(f"Contains '<|channel|>final': {'<|channel|>final' in decoded_prompt}")
print(f"Contains 'analysis channel': {'analysis channel' in decoded_prompt}")
print(f"Contains 'final channel': {'final channel' in decoded_prompt}")
print(f"Contains developer message content: {'ARC Puzzle Solver' in decoded_prompt}")

# Check what's in developer message
start_dev = decoded_prompt.find("<|start|>developer")
end_dev = decoded_prompt.find("<|end|>", start_dev)
if start_dev != -1 and end_dev != -1:
    dev_content = decoded_prompt[start_dev:end_dev+7]
    print(f"\nDeveloper message content:")
    print(dev_content)