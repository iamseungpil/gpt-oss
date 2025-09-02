#!/usr/bin/env python3
"""
Test GPT-OSS without final channel injection - let model switch naturally
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arc import train_problems
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format"""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

def parse_grid_from_response(response: str) -> np.ndarray:
    """Extract grid from response text"""
    lines = response.strip().split('\n')
    grid_rows = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip non-grid lines
        if any(skip in line.lower() for skip in ['grid:', 'output:', '```', 'channel', 'message', 'return', 'user', 'assistant']):
            continue
        
        parts = line.split()
        if parts:
            try:
                row = []
                for part in parts:
                    if part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
                        num = int(part)
                        if 0 <= num <= 9:
                            row.append(num)
                
                if row and len(row) > 1:  # Valid row should have multiple elements
                    grid_rows.append(row)
                elif grid_rows and len(row) != len(grid_rows[-1]):
                    # Stop if row length changes
                    break
                    
            except:
                if grid_rows:
                    break
                continue
    
    if grid_rows:
        # Check consistent row length
        row_lengths = [len(row) for row in grid_rows]
        if len(set(row_lengths)) == 1:
            return np.array(grid_rows)
    
    return None

def test_no_injection():
    """Test without forcing final channel"""
    
    print("🚀 Loading GPT-OSS model...")
    
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
    
    print("✅ Model loaded")
    print("="*70)
    
    # System setup
    system_setup = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

You are an expert at solving Abstract Reasoning Corpus (ARC) puzzles. You excel at identifying patterns, transformations, and rules in visual grids. 
Use the analysis channel for reasoning and the final channel for final answers.
When providing the final answer, switch to the final channel and output only the grid.<|end|>"""
    
    results = []
    
    # Test first problem only for quick testing
    problem = train_problems[0]
    print(f"🧩 Testing Problem: {problem.uid}")
    print("="*70)
    
    # Initialize conversation
    conversation = system_setup
    turn_responses = []
    
    # Prepare training examples
    examples_text = []
    for i, train_pair in enumerate(problem.train_pairs):
        examples_text.append(
            f"Example {i+1}:\nInput:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}"
        )
    
    all_examples = "\n\n".join(examples_text)
    test_input_grid = problem.test_pairs[0].x
    target_output_grid = problem.test_pairs[0].y
    
    # ========== TURN 1: Pattern Analysis ==========
    print("\n📝 TURN 1: Analyzing pattern")
    
    turn1_prompt = f"""<|start|>user<|message|>I have an ARC puzzle with {len(problem.train_pairs)} training examples:

{all_examples}

Please analyze and identify the transformation pattern or rule being applied from input to output.<|end|>"""
    
    conversation += turn1_prompt + "<|start|>assistant"
    
    # Generate Turn 1
    input_ids = tokenizer(conversation, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[200002],  # <|return|>
        )
    
    response1 = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    conversation += response1
    turn_responses.append({
        "turn": 1,
        "type": "pattern_analysis",
        "response": response1,
        "has_analysis_channel": "<|channel|>analysis" in response1
    })
    
    print(f"   ✓ Has analysis channel: {'<|channel|>analysis' in response1}")
    
    # ========== TURN 2: Verify Pattern ==========
    print("\n📝 TURN 2: Verifying pattern")
    
    turn2_prompt = f"""<|start|>user<|message|>Good analysis! Now verify your pattern with the training examples to make sure it correctly explains all transformations.<|end|>"""
    
    conversation += turn2_prompt + "<|start|>assistant"
    
    # Generate Turn 2
    input_ids = tokenizer(conversation, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[200002],
        )
    
    response2 = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    conversation += response2
    turn_responses.append({
        "turn": 2,
        "type": "pattern_verification",
        "response": response2,
        "has_analysis_channel": "<|channel|>analysis" in response2
    })
    
    print(f"   ✓ Has analysis channel: {'<|channel|>analysis' in response2}")
    
    # ========== FINAL TURN: Apply and Output (NO INJECTION) ==========
    print("\n📝 FINAL TURN: Apply pattern and output grid (no injection)")
    
    # More explicit instruction to use final channel
    final_prompt = f"""<|start|>user<|message|>Perfect! Now apply your verified pattern to this test input:

Test Input:
{grid_to_string(test_input_grid)}

Based on the pattern you identified (output size = input_length × non_zero_count, with input sequence repeated on diagonal), generate the output grid.

Remember to switch to the final channel for your answer and provide only the grid numbers.<|end|>"""
    
    # NO INJECTION - just add assistant start
    conversation += final_prompt + "<|start|>assistant"
    
    # Generate Final (let model choose channel)
    input_ids = tokenizer(conversation, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=2048,  # More tokens for full grid
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[200002],
        )
    
    final_response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    conversation += final_response
    turn_responses.append({
        "turn": 3,
        "type": "final_output",
        "response": final_response,
        "has_analysis": "<|channel|>analysis" in final_response,
        "has_final": "<|channel|>final" in final_response,
        "has_return": "<|return|>" in final_response
    })
    
    print(f"   ✓ Has analysis channel: {'<|channel|>analysis' in final_response}")
    print(f"   ✓ Has final channel: {'<|channel|>final' in final_response}")
    print(f"   ✓ Has return token: {'<|return|>' in final_response}")
    
    # Try to extract grid from response
    extracted_grid = parse_grid_from_response(final_response)
    
    if extracted_grid is not None:
        print(f"   ✅ Extracted grid shape: {extracted_grid.shape}")
        
        if extracted_grid.shape == target_output_grid.shape:
            accuracy = np.mean(extracted_grid == target_output_grid)
            print(f"   📊 Accuracy: {accuracy:.1%}")
        else:
            print(f"   ⚠️ Shape mismatch: expected {target_output_grid.shape}")
            accuracy = 0.0
    else:
        print(f"   ❌ Could not extract grid from response")
        accuracy = 0.0
        extracted_grid = None
    
    # Store results
    problem_result = {
        "problem_id": problem.uid,
        "test_input": test_input_grid.tolist(),
        "target_output": target_output_grid.tolist(),
        "model_output": extracted_grid.tolist() if extracted_grid is not None else None,
        "accuracy": float(accuracy),
        "channels_used": {
            "turn1_analysis": "<|channel|>analysis" in turn_responses[0]["response"],
            "turn2_analysis": "<|channel|>analysis" in turn_responses[1]["response"],
            "final_analysis": turn_responses[2]["has_analysis"],
            "final_final": turn_responses[2]["has_final"]
        },
        "final_response_length": len(final_response)
    }
    
    # Save conversation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_file = f"/tmp/no_injection_{problem.uid}_{timestamp}.txt"
    with open(conv_file, 'w') as f:
        f.write(f"NO INJECTION TEST\n")
        f.write(f"Problem: {problem.uid}\n")
        f.write(f"Accuracy: {accuracy:.1%}\n")
        f.write(f"Channels: analysis={problem_result['channels_used']['final_analysis']}, final={problem_result['channels_used']['final_final']}\n")
        f.write("="*70 + "\n\n")
        f.write(conversation)
        f.write("\n" + "="*70 + "\n")
        f.write("TARGET OUTPUT:\n")
        f.write(grid_to_string(target_output_grid))
        if extracted_grid is not None:
            f.write("\n" + "="*70 + "\n")
            f.write("MODEL OUTPUT:\n")
            f.write(grid_to_string(extracted_grid))
    
    print(f"\n💾 Saved to: {conv_file}")
    
    # Save JSON results
    json_file = f"/tmp/no_injection_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(problem_result, f, indent=2)
    
    print(f"📊 Results saved to: {json_file}")
    
    return conv_file, json_file

if __name__ == "__main__":
    conv_file, json_file = test_no_injection()