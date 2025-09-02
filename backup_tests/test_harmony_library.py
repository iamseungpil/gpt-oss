#!/usr/bin/env python3
"""
Test GPT-OSS with proper openai-harmony library for channel control
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arc import train_problems
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Import openai-harmony library
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

def test_harmony_library():
    """Test with proper openai-harmony library"""
    
    print("🚀 Loading GPT-OSS model and harmony encoding...")
    
    # Load harmony encoding
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    print("✅ Harmony encoding loaded")
    
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
    
    # Prepare pattern example
    example_pair = problem.train_pairs[0]
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
    
    # Test different reasoning levels with harmony library
    reasoning_levels = ["high", "medium", "low"]
    
    for reasoning_level in reasoning_levels:
        print(f"\n🧪 Testing reasoning level: {reasoning_level.upper()}")
        
        # Create conversation using harmony library
        try:
            # System message with reasoning level
            system_msg = Message.from_role_and_content(
                Role.SYSTEM,
                f"""You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final. Channel must be included for every message."""
            )
            
            # Developer message for strict instructions
            developer_msg = Message.from_role_and_content(
                Role.DEVELOPER,
                """# Instructions

You are solving ARC puzzles.

## CRITICAL REQUIREMENTS:
1. When generating a grid, use ONLY the final channel
2. Output ONLY grid numbers (space-separated, one row per line)
3. NO explanations, NO text, NO analysis in final channel
4. Grid format example:
   0 0 0 6 7
   0 0 6 7 8
   0 6 7 8 9

ALWAYS use final channel for grid output."""
            )
            
            # User message with pattern and request  
            user_msg = Message.from_role_and_content(
                Role.USER,
                f"""Generate the output grid using this EXACT pattern:

{pattern_example}

REQUIREMENTS:
- Output size: 20×20
- Use final channel ONLY
- Format: numbers separated by spaces, one row per line
- NO other text

Generate the grid NOW."""
            )
            
            # Create conversation
            conversation = Conversation.from_messages([
                system_msg,
                developer_msg, 
                user_msg
            ])
            
            print(f"   ✅ Conversation created with harmony library")
            
            # Render conversation for completion
            tokens = enc.render_conversation_for_completion(
                conversation,
                Role.ASSISTANT
            )
            
            # Do NOT manually inject final-channel tokens. Let the model switch channels naturally.
            print(f"   🎯 Rendered for completion (no forced final channel)")
            print(f"   📊 Token count: {len(tokens)}")
            
            # Convert tokens to tensor
            input_ids = torch.tensor([tokens], dtype=torch.long).to(model.device)
            
            # Generate response
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
            
            # Decode response
            response_tokens = output[0][input_ids.shape[1]:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=False)
            
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
                "model_output": extracted_grid.tolist() if extracted_grid is not None else None,
                "harmony_library_used": True,
                "forced_final_channel": False
            }
            
        except Exception as e:
            print(f"   ❌ Error with reasoning level {reasoning_level}: {str(e)}")
            results[reasoning_level] = {
                "reasoning_level": reasoning_level,
                "error": str(e),
                "harmony_library_used": True,
                "forced_final_channel": False
            }
    
    # Compile final results
    problem_result = {
        "problem_id": problem.uid,
        "test_input": test_input_grid.tolist(),
        "target_output": target_output_grid.tolist(),
        "library_used": "openai-harmony",
        "pattern_example_used": pattern_example,
        "reasoning_level_results": results,
        "summary": {
            "final_channel_achieved": {level: results[level].get("channels", {}).get("has_final", False) for level in reasoning_levels if "error" not in results[level]},
            "grid_extraction_success": {level: results[level].get("grid_extraction", {}).get("success", False) for level in reasoning_levels if "error" not in results[level]},
            "accuracy_scores": {level: results[level].get("grid_extraction", {}).get("accuracy", 0.0) for level in reasoning_levels if "error" not in results[level]}
        }
    }
    
    # Save detailed conversation log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_file = f"/tmp/harmony_library_{problem.uid}_{timestamp}.txt"
    with open(conv_file, 'w') as f:
        f.write(f"OPENAI-HARMONY LIBRARY TEST\n")
        f.write(f"Problem: {problem.uid}\n")
        f.write(f"Library: openai-harmony with forced final channel\n")
        f.write("="*70 + "\n\n")
        f.write(f"PATTERN EXAMPLE USED:\n{pattern_example}\n")
        f.write("="*70 + "\n\n")
        
        for level in reasoning_levels:
            result = results[level]
            if "error" not in result:
                f.write(f"REASONING LEVEL: {level.upper()}\n")
                f.write(f"Harmony library: {result.get('harmony_library_used', False)}\n")
                f.write(f"Forced final channel: {result.get('forced_final_channel', False)}\n")
                f.write(f"Channels - Analysis: {result['channels']['has_analysis']}, Final: {result['channels']['has_final']}\n")
                f.write(f"Grid extraction: {result['grid_extraction']['success']}, Accuracy: {result['grid_extraction']['accuracy']:.1%}\n")
                f.write("-"*50 + "\n")
                f.write(result['response'])
                f.write("\n" + "="*50 + "\n\n")
            else:
                f.write(f"REASONING LEVEL: {level.upper()} - ERROR\n")
                f.write(f"Error: {result['error']}\n")
                f.write("="*50 + "\n\n")
        
        f.write("TARGET OUTPUT:\n")
        f.write(grid_to_string(target_output_grid))
    
    print(f"\n💾 Saved to: {conv_file}")
    
    # Save JSON results
    json_file = f"/tmp/harmony_library_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(problem_result, f, indent=2)
    
    print(f"📊 Results saved to: {json_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 HARMONY LIBRARY RESULTS")
    print("="*70)
    for level in reasoning_levels:
        result = results[level]
        if "error" not in result:
            channels = result['channels']
            extraction = result['grid_extraction']
            
            print(f"{level.upper():>8}: Final={channels['has_final']:>5} | Grid={extraction['success']:>5} | Accuracy={extraction['accuracy']:>6.1%}")
        else:
            print(f"{level.upper():>8}: ERROR - {result['error']}")
    
    return conv_file, json_file

if __name__ == "__main__":
    conv_file, json_file = test_harmony_library()
