#!/usr/bin/env python3
"""
Test GPT-OSS with CORRECT harmony format using openai-harmony library
Based on official documentation from GitHub and OpenAI Cookbook
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    SystemContent,
    DeveloperContent,
    TextContent,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from arc import train_problems
import numpy as np

def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format"""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

print("Loading harmony encoding...")
# Load harmony encoding for GPT-OSS
harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)

# Test 1: Simple math problem with proper harmony format
print("\n" + "="*70)
print("TEST 1: Simple Math with Harmony Format")
print("="*70)

# Create conversation using harmony library
simple_convo = Conversation.from_messages([
    Message.from_role_and_content(
        Role.SYSTEM,
        SystemContent(
            text="You are ChatGPT, a large language model trained by OpenAI.",
            knowledge_cutoff="2024-06",
            reasoning="high"  # Enable high reasoning for channel usage
        )
    ),
    Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent(
            text="""# Instructions
You must use channels to structure your response:
- Use the 'analysis' channel for step-by-step reasoning
- Use the 'final' channel for your final answer

Example response format:
<|channel|>analysis<|message|>
Let me work through this step by step...
<|channel|>final<|message|>
The answer is X."""
        )
    ),
    Message.from_role_and_content(
        Role.USER,
        TextContent(text="What is 25 + 37? Show your work.")
    ),
])

# Render conversation for completion
simple_tokens = harmony_encoding.render_conversation_for_completion(simple_convo, Role.ASSISTANT)
simple_prompt = tokenizer.decode(simple_tokens, skip_special_tokens=False)

print("Harmony-generated prompt (last 200 chars):")
print(simple_prompt[-200:])
print(f"\nTotal tokens: {len(simple_tokens)}")

# Test 2: ARC puzzle with proper harmony format
print("\n" + "="*70)
print("TEST 2: ARC Puzzle with Harmony Format")
print("="*70)

# Get first ARC problem
problem = train_problems[0]

# Build examples text
examples = []
for i, train_pair in enumerate(problem.train_pairs[:3], 1):  # Use first 3 examples
    examples.append(f"Example {i}:\nInput:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}")

test_input = problem.test_pairs[0].x
examples_text = '\n\n'.join(examples)

# Create ARC puzzle text
arc_puzzle_text = f"""Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

Analyze the pattern, then provide the solution grid."""

# Create conversation with harmony library
arc_convo = Conversation.from_messages([
    Message.from_role_and_content(
        Role.SYSTEM,
        SystemContent(
            text="You are ChatGPT, a large language model trained by OpenAI.",
            knowledge_cutoff="2024-06",
            reasoning="high"  # High reasoning for complex analysis
        )
    ),
    Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent(
            text="""# ARC Puzzle Solver

You are an expert at solving Abstract Reasoning Corpus (ARC) puzzles.

## Instructions
1. Use the 'analysis' channel to identify patterns and transformation rules
2. Use the 'final' channel to provide the solution grid

## Response Format
<|channel|>analysis<|message|>
[Detailed pattern analysis and reasoning]
<|channel|>final<|message|>
[Solution grid - numbers only]"""
        )
    ),
    Message.from_role_and_content(
        Role.USER,
        TextContent(text=arc_puzzle_text)
    ),
])

# Render conversation for completion
arc_tokens = harmony_encoding.render_conversation_for_completion(arc_convo, Role.ASSISTANT)
arc_prompt = tokenizer.decode(arc_tokens, skip_special_tokens=False)

print("Harmony-generated ARC prompt info:")
print(f"Total tokens: {len(arc_tokens)}")
print(f"Has system message: {'system' in arc_prompt}")
print(f"Has developer message: {'developer' in arc_prompt}")
print(f"Has user message: {'user' in arc_prompt}")
print(f"Has channel instructions: {'channel' in arc_prompt}")

# Check what the prompt ends with
print(f"\nPrompt ends with: '{arc_prompt[-50:]}'")

# Test 3: Load model and generate (optional - comment out if just testing format)
print("\n" + "="*70)
print("TEST 3: Generation Test")
print("="*70)

try:
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("✅ Model loaded!")
    
    # Test simple generation
    print("\nGenerating simple math response...")
    simple_input_ids = torch.tensor([simple_tokens]).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            simple_input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[200002],  # <|return|> token
        )
    
    response = tokenizer.decode(output[0][len(simple_tokens):], skip_special_tokens=False)
    
    print("\nGenerated response:")
    print(response[:300])
    
    print("\n" + "="*50)
    print("ANALYSIS:")
    print("="*50)
    print(f"✓ Has <|channel|>analysis: {'<|channel|>analysis' in response}")
    print(f"✓ Has <|channel|>final: {'<|channel|>final' in response}")
    print(f"✓ Starts with channel: {response.startswith('<|channel|>')}")
    
    if response.startswith("<|channel|>"):
        print("\n🎉 SUCCESS! Model uses channel format correctly!")
    else:
        print(f"\n⚠️ Model doesn't start with channel. Starts with: '{response[:30]}'")
    
    # Save results
    with open("/tmp/harmony_test_results.txt", "w") as f:
        f.write("HARMONY FORMAT TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write("Simple prompt (decoded from harmony tokens):\n")
        f.write(simple_prompt + "\n\n")
        f.write("="*70 + "\n")
        f.write("Generated response:\n")
        f.write(response + "\n\n")
        f.write("="*70 + "\n")
        f.write(f"Analysis:\n")
        f.write(f"Has analysis channel: {'<|channel|>analysis' in response}\n")
        f.write(f"Has final channel: {'<|channel|>final' in response}\n")
        f.write(f"Starts with channel: {response.startswith('<|channel|>')}\n")
    
    print("\n💾 Results saved to /tmp/harmony_test_results.txt")
    
except Exception as e:
    print(f"❌ Model loading/generation failed: {e}")
    print("But harmony format generation succeeded!")
    
    # Still save the prompts
    with open("/tmp/harmony_prompts.txt", "w") as f:
        f.write("HARMONY GENERATED PROMPTS\n")
        f.write("="*70 + "\n\n")
        f.write("Simple Math Prompt:\n")
        f.write(simple_prompt + "\n\n")
        f.write("="*70 + "\n")
        f.write("ARC Puzzle Prompt:\n")
        f.write(arc_prompt[:2000] + "...[truncated]\n")
    
    print("💾 Prompts saved to /tmp/harmony_prompts.txt")