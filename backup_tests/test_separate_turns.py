#!/usr/bin/env python3
"""
Test GPT-OSS with separate turns: 1) Pattern analysis 2) Grid generation
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
        if any(skip in line.lower() for skip in ['grid:', 'output:', '```', 'channel', 'message', 'return', 'user', 'assistant', 'analysis', 'pattern', 'example']):
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

def test_separate_turns():
    """Test with separate pattern analysis and grid generation turns"""
    
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
Use the analysis channel for reasoning and step-by-step analysis.
When asked to generate a grid, provide the numerical grid output.<|end|>"""
    
    # Test first problem
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
    
    # ========== TURN 1: Pattern Analysis ONLY ==========
    print("\n📝 TURN 1: Pattern analysis only")
    
    turn1_prompt = f"""<|start|>user<|message|>I have an ARC puzzle with {len(problem.train_pairs)} training examples:

{all_examples}

Please analyze and identify the transformation pattern or rule being applied from input to output. Focus only on understanding the pattern - don't generate any grids yet.<|end|>"""
    
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
        "type": "pattern_analysis_only",
        "response": response1,
        "has_analysis_channel": "<|channel|>analysis" in response1,
        "has_final_channel": "<|channel|>final" in response1
    })
    
    print(f"   ✓ Has analysis channel: {'<|channel|>analysis' in response1}")
    print(f"   ✓ Has final channel: {'<|channel|>final' in response1}")
    
    # ========== TURN 2: Grid Generation Request ==========
    print("\n📝 TURN 2: Grid generation request")
    
    turn2_prompt = f"""<|start|>user<|message|>Great analysis! Now apply that pattern to generate the output grid for this test input:

Test Input:
{grid_to_string(test_input_grid)}

Please generate the complete output grid based on your pattern analysis. Provide the grid as numbers separated by spaces, one row per line.<|end|>"""
    
    conversation += turn2_prompt + "<|start|>assistant"
    
    # Generate Turn 2
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
    
    response2 = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    conversation += response2
    turn_responses.append({
        "turn": 2,
        "type": "grid_generation",
        "response": response2,
        "has_analysis_channel": "<|channel|>analysis" in response2,
        "has_final_channel": "<|channel|>final" in response2,
        "has_return": "<|return|>" in response2
    })
    
    print(f"   ✓ Has analysis channel: {'<|channel|>analysis' in response2}")
    print(f"   ✓ Has final channel: {'<|channel|>final' in response2}")
    print(f"   ✓ Has return token: {'<|return|>' in response2}")
    
    # Try to extract grid from response
    extracted_grid = parse_grid_from_response(response2)
    
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
            "turn1_analysis": turn_responses[0]["has_analysis_channel"],
            "turn1_final": turn_responses[0]["has_final_channel"],
            "turn2_analysis": turn_responses[1]["has_analysis_channel"],
            "turn2_final": turn_responses[1]["has_final_channel"],
        },
        "turn_responses": turn_responses
    }
    
    # Save conversation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_file = f"/tmp/separate_turns_{problem.uid}_{timestamp}.txt"
    with open(conv_file, 'w') as f:
        f.write(f"SEPARATE TURNS TEST\n")
        f.write(f"Problem: {problem.uid}\n")
        f.write(f"Accuracy: {accuracy:.1%}\n")
        f.write(f"Turn 1 - Analysis: {problem_result['channels_used']['turn1_analysis']}, Final: {problem_result['channels_used']['turn1_final']}\n")
        f.write(f"Turn 2 - Analysis: {problem_result['channels_used']['turn2_analysis']}, Final: {problem_result['channels_used']['turn2_final']}\n")
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
    json_file = f"/tmp/separate_turns_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(problem_result, f, indent=2)
    
    print(f"📊 Results saved to: {json_file}")
    
    return conv_file, json_file

if __name__ == "__main__":
    conv_file, json_file = test_separate_turns()