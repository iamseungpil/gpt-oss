#!/usr/bin/env python3
"""
Test pattern retention with explicit grid examples across all reasoning levels
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

def test_pattern_retention():
    """Test pattern retention with explicit examples across reasoning levels"""
    
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
    
    # Test first problem
    problem = train_problems[0]
    print(f"🧩 Testing Problem: {problem.uid}")
    print("="*70)
    
    test_input_grid = problem.test_pairs[0].x
    target_output_grid = problem.test_pairs[0].y
    
    # Prepare explicit pattern example from training data
    example_pair = problem.train_pairs[0]  # First training example
    pattern_example = f"""
PATTERN RULE: Grid size = (non-zero count × 5). Place input row at bottom-left, create diagonals for each non-zero element.

Example:
Input: {grid_to_string(example_pair.x)}
Output:
{grid_to_string(example_pair.y)}

Test Input: {grid_to_string(test_input_grid)}
Expected Output Size: 20×20 (4 non-zero elements × 5)
"""
    
    results = {}
    
    # Test all three reasoning levels
    reasoning_levels = ["high", "medium", "low"]
    
    for reasoning_level in reasoning_levels:
        print(f"\n🧪 Testing reasoning level: {reasoning_level.upper()}")
        
        # System setup with explicit pattern instruction
        system_setup = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final. Channel must be included for every message.
# CRITICAL: When asked for grid output, use ONLY the final channel and output ONLY grid numbers.<|end|><|start|>developer<|message|># Instructions

You are solving ARC puzzles. 

## STRICT OUTPUT REQUIREMENTS:
1. When generating a grid, IMMEDIATELY switch to final channel
2. Output ONLY the grid numbers (space-separated, one row per line)
3. NO explanations, NO text, NO analysis in final channel
4. Grid format example:
   0 0 0 6 7
   0 0 6 7 8
   0 6 7 8 9

Follow this pattern EXACTLY.<|end|>"""
        
        prompt = f"""<|start|>user<|message|>Generate the output grid using this EXACT pattern:

{pattern_example}

REQUIREMENTS:
- Output size: 20×20 
- Use final channel ONLY
- Format: numbers separated by spaces, one row per line
- NO other text

Generate the grid NOW.<|end|>"""
        
        conversation = system_setup + prompt + "<|start|>assistant"
        
        # Generate response
        input_ids = tokenizer(conversation, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[200002],  # <|return|>
            )
        
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
        
        # Analyze response
        has_analysis = "<|channel|>analysis" in response
        has_final = "<|channel|>final" in response
        has_return = "<|return|>" in response
        
        print(f"   ✓ Has analysis channel: {has_analysis}")
        print(f"   ✓ Has final channel: {has_final}")
        print(f"   ✓ Has return token: {has_return}")
        
        # Try to extract grid
        extracted_grid = parse_grid_from_response(response)
        
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
        results[reasoning_level] = {
            "reasoning_level": reasoning_level,
            "response": response,
            "channels": {
                "has_analysis": has_analysis,
                "has_final": has_final,
                "has_return": has_return
            },
            "grid_extraction": {
                "success": extracted_grid is not None,
                "shape": extracted_grid.shape if extracted_grid is not None else None,
                "accuracy": float(accuracy)
            },
            "model_output": extracted_grid.tolist() if extracted_grid is not None else None
        }
    
    # Compile final results
    problem_result = {
        "problem_id": problem.uid,
        "test_input": test_input_grid.tolist(),
        "target_output": target_output_grid.tolist(),
        "pattern_example_used": pattern_example,
        "reasoning_level_results": results,
        "summary": {
            "final_channel_achieved": {level: results[level]["channels"]["has_final"] for level in reasoning_levels},
            "grid_extraction_success": {level: results[level]["grid_extraction"]["success"] for level in reasoning_levels},
            "accuracy_scores": {level: results[level]["grid_extraction"]["accuracy"] for level in reasoning_levels}
        }
    }
    
    # Save detailed conversation log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_file = f"/tmp/pattern_retention_{problem.uid}_{timestamp}.txt"
    with open(conv_file, 'w') as f:
        f.write(f"PATTERN RETENTION TEST\n")
        f.write(f"Problem: {problem.uid}\n")
        f.write(f"Testing explicit pattern with in-context example\n")
        f.write("="*70 + "\n\n")
        f.write(f"PATTERN EXAMPLE USED:\n{pattern_example}\n")
        f.write("="*70 + "\n\n")
        
        for level in reasoning_levels:
            result = results[level]
            f.write(f"REASONING LEVEL: {level.upper()}\n")
            f.write(f"Channels - Analysis: {result['channels']['has_analysis']}, Final: {result['channels']['has_final']}\n")
            f.write(f"Grid extraction: {result['grid_extraction']['success']}, Accuracy: {result['grid_extraction']['accuracy']:.1%}\n")
            f.write("-"*50 + "\n")
            f.write(result['response'])
            f.write("\n" + "="*50 + "\n\n")
        
        f.write("TARGET OUTPUT:\n")
        f.write(grid_to_string(target_output_grid))
    
    print(f"\n💾 Saved to: {conv_file}")
    
    # Save JSON results
    json_file = f"/tmp/pattern_retention_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(problem_result, f, indent=2)
    
    print(f"📊 Results saved to: {json_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 SUMMARY RESULTS")
    print("="*70)
    for level in reasoning_levels:
        result = results[level]
        channels = result['channels']
        extraction = result['grid_extraction']
        
        print(f"{level.upper():>8}: Final={channels['has_final']:>5} | Grid={extraction['success']:>5} | Accuracy={extraction['accuracy']:>6.1%}")
    
    return conv_file, json_file

if __name__ == "__main__":
    conv_file, json_file = test_pattern_retention()