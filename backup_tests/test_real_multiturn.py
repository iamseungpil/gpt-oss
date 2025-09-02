#!/usr/bin/env python3
"""
Real multi-turn interaction with GPT-OSS for ARC tasks
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arc import train_problems
import numpy as np

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format"""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

def test_real_multiturn():
    """Test real multi-turn interaction"""
    
    print("🚀 Loading GPT-OSS model...")
    
    # Load tokenizer and model
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
    
    print("✅ Model loaded")
    print("="*70)
    
    # Get first ARC problem
    problem = train_problems[0]
    print(f"🧩 Testing ARC Problem: {problem.uid}")
    
    # Build examples
    examples = []
    for i, train_pair in enumerate(problem.train_pairs[:2], 1):  # Use only 2 for speed
        examples.append(f"Example {i}:\nInput:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}")
    
    test_input = problem.test_pairs[0].x
    examples_text = '\n\n'.join(examples)
    
    # ========== TURN 1: Ask for analysis ==========
    print("\n" + "="*70)
    print("📝 TURN 1: Asking for pattern analysis")
    print("="*70)
    
    turn1_prompt = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

You are an expert at solving ARC puzzles. Use the analysis channel for reasoning and the final channel for answers.<|end|><|start|>user<|message|>I have an ARC puzzle. Here are the examples:

{examples_text}

Can you analyze what pattern or transformation rule is being applied here?<|end|><|start|>assistant"""
    
    # Generate first response
    input_ids = tokenizer(turn1_prompt, return_tensors="pt").input_ids.to(model.device)
    
    print("🔄 Generating Turn 1 response...")
    with torch.no_grad():
        output1 = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[200002],  # <|return|> token
        )
    
    response1 = tokenizer.decode(output1[0][input_ids.shape[1]:], skip_special_tokens=False)
    print(f"\n📤 Turn 1 Response (first 400 chars):")
    print(response1[:400])
    
    # Check channels in Turn 1
    print(f"\n📊 Turn 1 Analysis:")
    print(f"   - Has analysis channel: {'<|channel|>analysis' in response1}")
    print(f"   - Has final channel: {'<|channel|>final' in response1}")
    print(f"   - Has return token: {'<|return|>' in response1}")
    
    # ========== TURN 2: Ask for solution ==========
    print("\n" + "="*70)
    print("📝 TURN 2: Asking for the solution")
    print("="*70)
    
    # Build conversation history for Turn 2
    # Include the full conversation so far
    turn2_prompt = turn1_prompt + response1
    
    # Add second user message
    turn2_prompt += f"""<|start|>user<|message|>Good analysis! Now here's the test input:

{grid_to_string(test_input)}

Based on your analysis of the pattern, what should the output grid be? Please provide only the grid with numbers.<|end|><|start|>assistant"""
    
    # Generate second response
    input_ids2 = tokenizer(turn2_prompt, return_tensors="pt").input_ids.to(model.device)
    
    print("🔄 Generating Turn 2 response...")
    with torch.no_grad():
        output2 = model.generate(
            input_ids2,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[200002],  # <|return|> token
        )
    
    response2 = tokenizer.decode(output2[0][input_ids2.shape[1]:], skip_special_tokens=False)
    print(f"\n📤 Turn 2 Response (first 500 chars):")
    print(response2[:500])
    
    # Check channels in Turn 2
    print(f"\n📊 Turn 2 Analysis:")
    print(f"   - Has analysis channel: {'<|channel|>analysis' in response2}")
    print(f"   - Has final channel: {'<|channel|>final' in response2}")
    print(f"   - Has return token: {'<|return|>' in response2}")
    
    # ========== TURN 3: Force final channel ==========
    print("\n" + "="*70)
    print("📝 TURN 3: Explicitly asking for final answer")
    print("="*70)
    
    # Build conversation with both turns
    turn3_prompt = turn2_prompt + response2
    
    # Add third user message explicitly asking for final channel
    turn3_prompt += f"""<|start|>user<|message|>Please provide your final answer with just the output grid.<|end|><|start|>assistant<|channel|>final<|message|>"""
    
    # Generate third response (starting from final channel)
    input_ids3 = tokenizer(turn3_prompt, return_tensors="pt").input_ids.to(model.device)
    
    print("🔄 Generating Turn 3 response (pre-injected final channel)...")
    with torch.no_grad():
        output3 = model.generate(
            input_ids3,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[200002],  # <|return|> token
        )
    
    response3 = tokenizer.decode(output3[0][input_ids3.shape[1]:], skip_special_tokens=False)
    print(f"\n📤 Turn 3 Response (pre-injected final):")
    print(response3[:500])
    
    # Check if we got a grid
    print(f"\n📊 Turn 3 Analysis:")
    print(f"   - Has return token: {'<|return|>' in response3}")
    print(f"   - Response length: {len(response3)} chars")
    
    # Save all responses
    filename = f"/tmp/real_multiturn_{problem.uid}.txt"
    with open(filename, 'w') as f:
        f.write("REAL MULTI-TURN INTERACTION\n")
        f.write("="*70 + "\n\n")
        
        f.write("TURN 1 - Pattern Analysis Request:\n")
        f.write(turn1_prompt + "\n")
        f.write("\nTURN 1 - Response:\n")
        f.write(response1 + "\n")
        f.write("="*70 + "\n\n")
        
        f.write("TURN 2 - Solution Request:\n")
        f.write("User: Based on your analysis, what should the output grid be?\n")
        f.write("\nTURN 2 - Response:\n")
        f.write(response2 + "\n")
        f.write("="*70 + "\n\n")
        
        f.write("TURN 3 - Final Answer Request (pre-injected final channel):\n")
        f.write("\nTURN 3 - Response:\n")
        f.write(response3 + "\n")
    
    print(f"\n💾 All turns saved to: {filename}")
    print("\n✅ Test completed!")
    print("\nSummary:")
    print("- Turn 1: Asked for analysis")
    print("- Turn 2: Asked for solution based on analysis")
    print("- Turn 3: Pre-injected final channel to force final answer")

if __name__ == "__main__":
    test_real_multiturn()