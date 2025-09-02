#!/usr/bin/env python3
"""
Improved multi-turn test for GPT-OSS with ARC tasks
- Step 1: Analyze pattern
- Step 2: Verify pattern with training examples
- Step 3: Apply pattern to test input
- Final: Output grid only
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
    """Try to extract grid from response"""
    lines = response.strip().split('\n')
    grid_rows = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip non-grid lines
        if any(skip in line.lower() for skip in ['grid:', 'output:', '```', 'channel', 'message', 'return']):
            continue
        
        parts = line.split()
        if parts:
            try:
                row = [int(p) for p in parts if p.isdigit() or (p.startswith('-') and p[1:].isdigit())]
                if row:
                    grid_rows.append(row)
            except:
                continue
    
    if grid_rows:
        # Check consistent row length
        row_lengths = [len(row) for row in grid_rows]
        if len(set(row_lengths)) == 1:
            return np.array(grid_rows)
    
    return None

def test_multiturn_improved():
    """Run improved multi-turn test"""
    
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
When asked for output grids, provide ONLY the grid numbers without any explanation.<|end|>"""
    
    # Test first 3 ARC problems
    results = []
    
    for prob_idx in range(3):
        problem = train_problems[prob_idx]
        print(f"\n{'='*70}")
        print(f"🧩 Testing Problem {prob_idx + 1}: {problem.uid}")
        print(f"   Training examples: {len(problem.train_pairs)}")
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
        
        print(f"   ✓ Response length: {len(response1)} chars")
        print(f"   ✓ Has analysis channel: {'<|channel|>analysis' in response1}")
        
        # ========== TURN 2: Verify Pattern ==========
        print("\n📝 TURN 2: Verifying pattern")
        
        turn2_prompt = f"""<|start|>user<|message|>Good analysis! Now let's verify your identified pattern. 

Look at the training examples again and confirm whether your pattern correctly explains the transformation from each input to its corresponding output. 

Are there any adjustments needed to your pattern?<|end|>"""
        
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
        
        print(f"   ✓ Response length: {len(response2)} chars")
        print(f"   ✓ Has analysis channel: {'<|channel|>analysis' in response2}")
        
        # ========== TURN 3: Apply to Test Input ==========
        print("\n📝 TURN 3: Applying pattern to test input")
        
        turn3_prompt = f"""<|start|>user<|message|>Perfect! Now apply your verified pattern to this test input:

Test Input:
{grid_to_string(test_input_grid)}

Based on the pattern you identified and verified, what should the output grid be? Think through how the transformation applies to this specific input.<|end|>"""
        
        conversation += turn3_prompt + "<|start|>assistant"
        
        # Generate Turn 3
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
        
        response3 = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
        conversation += response3
        turn_responses.append({
            "turn": 3,
            "type": "pattern_application",
            "response": response3,
            "has_analysis_channel": "<|channel|>analysis" in response3
        })
        
        print(f"   ✓ Response length: {len(response3)} chars")
        print(f"   ✓ Has analysis channel: {'<|channel|>analysis' in response3}")
        
        # ========== FINAL TURN: Output Grid Only ==========
        print("\n📝 FINAL TURN: Getting final output grid")
        
        final_prompt = f"""<|start|>user<|message|>Now provide ONLY the final output grid. No explanation, just the grid numbers.<|end|>"""
        
        # Force final channel
        conversation += final_prompt + "<|start|>assistant<|channel|>final<|message|>"
        
        # Generate Final
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
        
        final_response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
        conversation += final_response
        turn_responses.append({
            "turn": 4,
            "type": "final_output",
            "response": final_response,
            "has_return": "<|return|>" in final_response
        })
        
        print(f"   ✓ Response length: {len(final_response)} chars")
        print(f"   ✓ Has return token: {'<|return|>' in final_response}")
        
        # Try to extract grid
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
            print(f"   ❌ Could not extract grid")
            accuracy = 0.0
            extracted_grid = None
        
        # Store results
        problem_result = {
            "problem_id": problem.uid,
            "problem_index": prob_idx,
            "training_examples": len(problem.train_pairs),
            "test_input_shape": test_input_grid.shape.tolist() if hasattr(test_input_grid.shape, 'tolist') else list(test_input_grid.shape),
            "target_output_shape": target_output_grid.shape.tolist() if hasattr(target_output_grid.shape, 'tolist') else list(target_output_grid.shape),
            "turns": turn_responses,
            "extracted_grid": extracted_grid.tolist() if extracted_grid is not None else None,
            "target_grid": target_output_grid.tolist() if hasattr(target_output_grid, 'tolist') else target_output_grid,
            "accuracy": float(accuracy),
            "full_conversation_length": len(conversation)
        }
        
        results.append(problem_result)
        
        # Save individual conversation
        conv_file = f"/tmp/improved_multiturn_{problem.uid}.txt"
        with open(conv_file, 'w') as f:
            f.write(f"IMPROVED MULTI-TURN CONVERSATION\n")
            f.write(f"Problem: {problem.uid}\n")
            f.write(f"Accuracy: {accuracy:.1%}\n")
            f.write("="*70 + "\n\n")
            f.write(conversation)
            f.write("\n" + "="*70 + "\n")
            f.write("TARGET OUTPUT:\n")
            f.write(grid_to_string(target_output_grid))
        
        print(f"   💾 Saved conversation to: {conv_file}")
    
    # Save all results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"/tmp/multiturn_results_{timestamp}.json"
    
    with open(json_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "model": "openai/gpt-oss-20b",
            "num_problems": len(results),
            "average_accuracy": np.mean([r["accuracy"] for r in results]),
            "problems": results
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ All tests completed!")
    print(f"📊 Average accuracy: {np.mean([r['accuracy'] for r in results]):.1%}")
    print(f"💾 Results saved to: {json_file}")
    
    return json_file

if __name__ == "__main__":
    json_file = test_multiturn_improved()
    
    # Visualize results
    print("\n🎨 Visualizing grids...")
    from visualize_arc_grids import visualize_multiturn_responses
    
    for i in range(3):
        problem = train_problems[i]
        conv_file = f"/tmp/improved_multiturn_{problem.uid}.txt"
        if os.path.exists(conv_file):
            output_path, grids = visualize_multiturn_responses(conv_file)
            if output_path:
                print(f"   ✓ Visualization saved: {output_path}")