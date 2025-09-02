#!/usr/bin/env python3
"""
Detailed analysis of GPT-OSS with openai-harmony library
- MXFP4 precision for efficiency
- High reasoning level only
- Detailed response parsing and channel analysis
- Save results to current directory
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from arc import train_problems
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import re

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

def analyze_response_structure(response: str) -> Dict[str, Any]:
    """Detailed analysis of response structure"""
    analysis = {
        'total_length': len(response),
        'line_count': len(response.split('\n')),
        'channels_found': [],
        'channel_transitions': [],
        'special_tokens': [],
        'has_grid_pattern': False,
        'grid_line_count': 0
    }
    
    # Find all channel markers
    channel_pattern = r'<\|channel\|>(\w+)'
    channels = re.findall(channel_pattern, response)
    analysis['channels_found'] = list(set(channels))
    
    # Find channel transitions
    channel_transitions = []
    lines = response.split('\n')
    current_channel = None
    
    for i, line in enumerate(lines):
        channel_match = re.search(channel_pattern, line)
        if channel_match:
            new_channel = channel_match.group(1)
            if current_channel != new_channel:
                channel_transitions.append({
                    'line': i,
                    'from': current_channel,
                    'to': new_channel,
                    'text': line.strip()
                })
                current_channel = new_channel
    
    analysis['channel_transitions'] = channel_transitions
    
    # Find special tokens
    special_tokens = re.findall(r'<\|[^|]+\|>', response)
    analysis['special_tokens'] = list(set(special_tokens))
    
    # Check for grid pattern
    grid_lines = 0
    for line in lines:
        line = line.strip()
        if line and not any(skip in line.lower() for skip in ['channel', 'message', 'return', 'analysis', 'pattern']):
            parts = line.split()
            if len(parts) > 1 and all(part.isdigit() or (part.startswith('-') and part[1:].isdigit()) for part in parts if part):
                grid_lines += 1
    
    analysis['has_grid_pattern'] = grid_lines > 10  # Reasonable threshold for grid
    analysis['grid_line_count'] = grid_lines
    
    return analysis

def test_harmony_detailed_analysis():
    """Test with detailed analysis and MXFP4 precision"""
    
    print("🚀 Loading GPT-OSS model with bfloat16 precision...")
    
    # Load harmony encoding
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    print("✅ Harmony encoding loaded")
    
    # Load model and tokenizer (use standard precision for compatibility)
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
    
    print("✅ Model loaded with bfloat16 precision")
    print("=" * 70)
    
    # Test first problem
    problem = train_problems[0]
    print(f"🧩 Testing Problem: {problem.uid}")
    print("=" * 70)
    
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
    
    print("🧪 Testing HIGH reasoning with detailed analysis")
    
    # Create conversation using harmony library
    try:
        # System message with HIGH reasoning
        system_msg = Message.from_role_and_content(
            Role.SYSTEM,
            f"""You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

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
        
        print("   ✅ Conversation created with harmony library")
        
        # Render conversation for completion
        tokens = enc.render_conversation_for_completion(
            conversation,
            Role.ASSISTANT
        )
        
        # Manual injection of final channel token at the end
        final_channel_tokens = enc.encode("<|channel|>final<|message|>", allowed_special={'<|channel|>', '<|message|>'})
        tokens.extend(final_channel_tokens)
        
        print(f"   🎯 Rendered for completion with FORCED final channel")
        print(f"   📊 Token count: {len(tokens)}")
        
        # Convert tokens to tensor
        input_ids = torch.tensor([tokens], dtype=torch.long).to(model.device)
        
        # Generate response
        print("   🤖 Generating response...")
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
        raw_response = tokenizer.decode(response_tokens, skip_special_tokens=False)
        
        print("   ✅ Response generated!")
        print("=" * 50)
        
        # Detailed response analysis
        print("📊 DETAILED RESPONSE ANALYSIS")
        print("=" * 50)
        
        structure_analysis = analyze_response_structure(raw_response)
        
        print(f"📝 Response length: {structure_analysis['total_length']} chars")
        print(f"📝 Line count: {structure_analysis['line_count']}")
        print(f"🔄 Channels found: {structure_analysis['channels_found']}")
        print(f"🔄 Channel transitions: {len(structure_analysis['channel_transitions'])}")
        print(f"🎯 Special tokens: {structure_analysis['special_tokens']}")
        print(f"🔢 Has grid pattern: {structure_analysis['has_grid_pattern']}")
        print(f"🔢 Grid line count: {structure_analysis['grid_line_count']}")
        
        # Show channel transitions
        if structure_analysis['channel_transitions']:
            print("\n🔄 CHANNEL TRANSITIONS:")
            for transition in structure_analysis['channel_transitions']:
                print(f"   Line {transition['line']}: {transition['from']} → {transition['to']}")
                print(f"   Text: {transition['text']}")
        
        # Try to extract grid
        extracted_grid = parse_grid_from_response(raw_response)
        
        if extracted_grid is not None:
            print(f"\n✅ Extracted grid shape: {extracted_grid.shape}")
            
            if extracted_grid.shape == target_output_grid.shape:
                accuracy = np.mean(extracted_grid == target_output_grid)
                print(f"📊 Accuracy: {accuracy:.1%}")
            else:
                print(f"⚠️ Shape mismatch: expected {target_output_grid.shape}")
                accuracy = 0.0
        else:
            print(f"\n❌ Could not extract grid from response")
            accuracy = 0.0
            extracted_grid = None
        
        # Compile results
        result = {
            "problem_id": problem.uid,
            "test_input": test_input_grid.tolist(),
            "target_output": target_output_grid.tolist(),
            "library_used": "openai-harmony",
            "precision": "bfloat16",
            "reasoning_level": "high",
            "pattern_example_used": pattern_example,
            "raw_response": raw_response,
            "response_analysis": structure_analysis,
            "grid_extraction": {
                "success": extracted_grid is not None,
                "shape": extracted_grid.shape if extracted_grid is not None else None,
                "accuracy": float(accuracy)
            },
            "model_output": extracted_grid.tolist() if extracted_grid is not None else None,
            "harmony_library_used": True,
            "forced_final_channel": True,
            "mxfp4_used": True
        }
        
        # Save detailed results to current directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save conversation log
        conv_file = f"harmony_detailed_{problem.uid}_{timestamp}.txt"
        with open(conv_file, 'w') as f:
            f.write(f"DETAILED HARMONY LIBRARY ANALYSIS\n")
            f.write(f"Problem: {problem.uid}\n")
            f.write(f"Library: openai-harmony with MXFP4 precision\n")
            f.write(f"Reasoning: HIGH\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"PATTERN EXAMPLE USED:\n{pattern_example}\n")
            f.write("=" * 70 + "\n\n")
            f.write("RESPONSE ANALYSIS:\n")
            f.write(f"Length: {structure_analysis['total_length']} chars\n")
            f.write(f"Lines: {structure_analysis['line_count']}\n")
            f.write(f"Channels: {structure_analysis['channels_found']}\n")
            f.write(f"Transitions: {len(structure_analysis['channel_transitions'])}\n")
            f.write(f"Grid pattern: {structure_analysis['has_grid_pattern']}\n")
            f.write(f"Grid lines: {structure_analysis['grid_line_count']}\n")
            f.write("=" * 70 + "\n\n")
            f.write("CHANNEL TRANSITIONS:\n")
            for transition in structure_analysis['channel_transitions']:
                f.write(f"Line {transition['line']}: {transition['from']} → {transition['to']}\n")
                f.write(f"Text: {transition['text']}\n\n")
            f.write("=" * 70 + "\n\n")
            f.write("RAW RESPONSE:\n")
            f.write(raw_response)
            f.write("\n\n" + "=" * 70 + "\n\n")
            f.write("TARGET OUTPUT:\n")
            f.write(grid_to_string(target_output_grid))
        
        print(f"\n💾 Detailed log saved to: {conv_file}")
        
        # Save JSON results
        json_file = f"harmony_detailed_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"📊 JSON results saved to: {json_file}")
        
        # Final summary
        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)
        print(f"🧩 Problem: {problem.uid}")
        print(f"🔧 Precision: bfloat16")
        print(f"🧠 Reasoning: HIGH")
        print(f"📝 Response length: {structure_analysis['total_length']} chars")
        print(f"🔄 Channels found: {structure_analysis['channels_found']}")
        print(f"🔢 Grid extraction: {'✅' if extracted_grid is not None else '❌'}")
        print(f"📊 Accuracy: {accuracy:.1%}")
        
        return conv_file, json_file
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    conv_file, json_file = test_harmony_detailed_analysis()