#!/usr/bin/env python3
"""
Test GPT-OSS with adaptive reasoning levels: high for analysis, medium for output
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

def test_adaptive_reasoning():
    """Test with adaptive reasoning levels: high for analysis, medium for output"""
    
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
    
    # ========== TURN 1: Deep Analysis with HIGH Reasoning ==========
    print("\n📝 TURN 1: Pattern analysis (reasoning: HIGH)")
    
    # System setup for analysis with HIGH reasoning
    system_analysis = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.
# Use analysis channel for deep pattern identification and rule discovery.<|end|><|start|>developer<|message|># Instructions

You are an expert at solving Abstract Reasoning Corpus (ARC) puzzles. 

Analyze the training examples thoroughly to identify the transformation pattern.
Use the analysis channel to work through the logic step-by-step.
Do not generate any grids yet - focus only on understanding the pattern.<|end|>"""
    
    turn1_prompt = f"""<|start|>user<|message|>Analyze this ARC puzzle with {len(problem.train_pairs)} training examples to identify the transformation pattern:

{all_examples}

Focus on discovering the exact rule that transforms input to output. Analyze:
1. Output grid size pattern
2. Element placement rules  
3. Diagonal or spatial arrangements

Be thorough in your analysis.<|end|>"""
    
    conversation1 = system_analysis + turn1_prompt + "<|start|>assistant"
    
    # Generate Turn 1 with HIGH reasoning
    input_ids = tokenizer(conversation1, return_tensors="pt").input_ids.to(model.device)
    
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
    
    response1 = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    turn_responses.append({
        "turn": 1,
        "reasoning_level": "high",
        "type": "pattern_analysis",
        "response": response1,
        "has_analysis_channel": "<|channel|>analysis" in response1,
        "has_final_channel": "<|channel|>final" in response1
    })
    
    print(f"   ✓ Has analysis channel: {'<|channel|>analysis' in response1}")
    print(f"   ✓ Has final channel: {'<|channel|>final' in response1}")
    
    # ========== TURN 2: Grid Generation with MEDIUM Reasoning ==========
    print("\n📝 TURN 2: Grid generation (reasoning: MEDIUM)")
    
    # System setup for output with MEDIUM reasoning
    system_output = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
# Use final channel for delivering completed results to users.
# When generating grids, switch to final channel and output only the grid numbers.<|end|><|start|>developer<|message|># Instructions

You have analyzed the ARC pattern. Now apply it to generate the output grid.

## Output Instructions:
1. Apply the discovered pattern to the test input
2. Switch to the final channel for your answer
3. Output the complete grid as numbers separated by spaces, one row per line
4. No explanations in final channel - just the grid numbers<|end|>"""
    
    turn2_prompt = f"""<|start|>user<|message|>Based on your analysis, apply the pattern to this test input and generate the output grid:

Test Input:
{grid_to_string(test_input_grid)}

Output the complete {target_output_grid.shape[0]}×{target_output_grid.shape[1]} grid using the pattern you identified.
Switch to the final channel and provide only the grid numbers.<|end|>"""
    
    conversation2 = system_output + turn2_prompt + "<|start|>assistant"
    
    # Generate Turn 2 with MEDIUM reasoning
    input_ids = tokenizer(conversation2, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[200002],
        )
    
    response2 = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    turn_responses.append({
        "turn": 2,
        "reasoning_level": "medium",
        "type": "grid_generation",
        "response": response2,
        "has_analysis_channel": "<|channel|>analysis" in response2,
        "has_final_channel": "<|channel|>final" in response2,
        "has_return": "<|return|>" in response2
    })
    
    print(f"   ✓ Has analysis channel: {'<|channel|>analysis' in response2}")
    print(f"   ✓ Has final channel: {'<|channel|>final' in response2}")
    print(f"   ✓ Has return token: {'<|return|>' in response2}")
    
    # ========== TURN 3: Final Push with LOW Reasoning if needed ==========
    final_response = response2
    if "<|channel|>final" not in response2:
        print("\n📝 TURN 3: Final push (reasoning: LOW)")
        
        # System setup with LOW reasoning for direct output
        system_final = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: low

# Valid channels: analysis, commentary, final. Channel must be included for every message.
# Output directly to final channel without extensive analysis.<|end|><|start|>developer<|message|># Instructions

Generate the grid output immediately. No analysis needed.
Use final channel only. Output format: numbers separated by spaces, one row per line.<|end|>"""
        
        turn3_prompt = f"""<|start|>user<|message|>Generate the {target_output_grid.shape[0]}×{target_output_grid.shape[1]} output grid for test input {grid_to_string(test_input_grid).replace(chr(10), ' ')}. 

Use the diagonal pattern with non-zero count × 5 size rule.
Final channel only - just the grid numbers.<|end|>"""
        
        conversation3 = system_final + turn3_prompt + "<|start|>assistant"
        
        # Generate Turn 3 with LOW reasoning
        input_ids = tokenizer(conversation3, return_tensors="pt").input_ids.to(model.device)
        
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
        
        response3 = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
        turn_responses.append({
            "turn": 3,
            "reasoning_level": "low",
            "type": "final_push",
            "response": response3,
            "has_analysis_channel": "<|channel|>analysis" in response3,
            "has_final_channel": "<|channel|>final" in response3,
            "has_return": "<|return|>" in response3
        })
        
        print(f"   ✓ Has analysis channel: {'<|channel|>analysis' in response3}")
        print(f"   ✓ Has final channel: {'<|channel|>final' in response3}")
        print(f"   ✓ Has return token: {'<|return|>' in response3}")
        
        final_response = response3
    
    # Try to extract grid from final response
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
        "adaptive_reasoning": {
            "turn1_level": "high",
            "turn2_level": "medium", 
            "turn3_level": "low" if len(turn_responses) > 2 else None
        },
        "channels_used": {
            "final_channel_achieved": any("<|channel|>final" in resp["response"] for resp in turn_responses),
            "final_channel_turns": [i+1 for i, resp in enumerate(turn_responses) if "<|channel|>final" in resp["response"]]
        },
        "turn_responses": turn_responses,
        "num_turns": len(turn_responses)
    }
    
    # Save conversation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_file = f"/tmp/adaptive_reasoning_{problem.uid}_{timestamp}.txt"
    with open(conv_file, 'w') as f:
        f.write(f"ADAPTIVE REASONING TEST\n")
        f.write(f"Problem: {problem.uid}\n")
        f.write(f"Accuracy: {accuracy:.1%}\n")
        f.write(f"Reasoning levels: Turn1=high, Turn2=medium, Turn3={problem_result['adaptive_reasoning']['turn3_level']}\n")
        f.write(f"Number of turns: {len(turn_responses)}\n")
        f.write(f"Final channel achieved: {problem_result['channels_used']['final_channel_achieved']}\n")
        f.write("="*70 + "\n\n")
        
        # Write each turn separately
        for i, resp in enumerate(turn_responses):
            f.write(f"TURN {i+1} (reasoning: {resp['reasoning_level']}):\n")
            f.write(resp['response'] + "\n")
            f.write("="*50 + "\n\n")
        
        f.write("TARGET OUTPUT:\n")
        f.write(grid_to_string(target_output_grid))
        if extracted_grid is not None:
            f.write("\n" + "="*70 + "\n")
            f.write("MODEL OUTPUT:\n")
            f.write(grid_to_string(extracted_grid))
    
    print(f"\n💾 Saved to: {conv_file}")
    
    # Save JSON results
    json_file = f"/tmp/adaptive_reasoning_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(problem_result, f, indent=2)
    
    print(f"📊 Results saved to: {json_file}")
    
    return conv_file, json_file

if __name__ == "__main__":
    conv_file, json_file = test_adaptive_reasoning()