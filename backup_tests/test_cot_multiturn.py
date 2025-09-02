#!/usr/bin/env python3
"""
CoT-style multi-turn test for ARC with step-by-step analysis
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

def test_cot_multiturn():
    """Test CoT-style multi-turn interaction"""
    
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
    print(f"   Train examples: {len(problem.train_pairs)}")
    print(f"   Test examples: {len(problem.test_pairs)}")
    print("="*70)
    
    # System and developer setup
    system_setup = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

You are an expert at solving ARC puzzles. Use the analysis channel for reasoning and the final channel for final answers. When asked for grids, provide ONLY the grid numbers without any explanation.<|end|>"""
    
    # Build conversation progressively
    conversation = system_setup
    all_responses = []
    
    # Prepare all examples
    examples_text = []
    for i, train_pair in enumerate(problem.train_pairs):
        examples_text.append({
            "index": i + 1,
            "input": grid_to_string(train_pair.x),
            "output": grid_to_string(train_pair.y)
        })
    
    # Test input
    test_input = grid_to_string(problem.test_pairs[0].x)
    
    # ========== TURN 1: Overall Pattern Analysis ==========
    print("\n📝 TURN 1: Analyzing overall pattern")
    print("-"*50)
    
    # Show all examples for pattern analysis
    examples_for_analysis = "\n\n".join([
        f"Example {ex['index']}:\nInput:\n{ex['input']}\nOutput:\n{ex['output']}"
        for ex in examples_text
    ])
    
    turn1_user = f"""<|start|>user<|message|>I have an ARC puzzle with {len(examples_text)} examples. Here they are:

{examples_for_analysis}

Please analyze the overall pattern or transformation rule being applied across all examples.<|end|>"""
    
    conversation += turn1_user + "<|start|>assistant"
    
    # Generate Turn 1
    input_ids = tokenizer(conversation, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[200002],  # <|return|>
        )
    
    response1 = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    all_responses.append(("Pattern Analysis", response1))
    conversation += response1
    
    print(f"Response preview: {response1[:200]}...")
    print(f"Has analysis channel: {'<|channel|>analysis' in response1}")
    
    # ========== TURNS 2 to N-1: Verify each training example ==========
    for idx, example in enumerate(examples_text):
        print(f"\n📝 TURN {idx+2}: Verifying Example {example['index']}")
        print("-"*50)
        
        turn_user = f"""<|start|>user<|message|>Based on the pattern you identified, let's verify Example {example['index']}.

Input:
{example['input']}

What should the output grid be? Provide ONLY the grid numbers.<|end|>"""
        
        conversation += turn_user + "<|start|>assistant"
        
        # Generate response
        input_ids = tokenizer(conversation, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[200002],
            )
        
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
        all_responses.append((f"Example {example['index']} Verification", response))
        conversation += response
        
        # Check if grid is in response
        has_grid = any(char.isdigit() for char in response)
        print(f"Response length: {len(response)} chars")
        print(f"Contains numbers (grid): {has_grid}")
        print(f"Has final channel: {'<|channel|>final' in response}")
    
    # ========== FINAL TURN: Test input ==========
    print(f"\n📝 TURN {len(examples_text)+2}: Final test input")
    print("-"*50)
    
    final_turn_user = f"""<|start|>user<|message|>Excellent! Now apply the same pattern to this test input:

Test Input:
{test_input}

Provide your final answer with ONLY the output grid.<|end|>"""
    
    # Force final channel
    conversation += final_turn_user + "<|start|>assistant<|channel|>final<|message|>"
    
    # Generate final response
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
    all_responses.append(("Final Test Output", final_response))
    conversation += final_response
    
    print(f"Final response preview: {final_response[:300]}...")
    print(f"Has return token: {'<|return|>' in final_response}")
    
    # ========== Save complete conversation ==========
    print("\n" + "="*70)
    print("💾 Saving complete conversation...")
    
    filename = f"/tmp/cot_multiturn_{problem.uid}.txt"
    with open(filename, 'w') as f:
        f.write("COT-STYLE MULTI-TURN CONVERSATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Problem: {problem.uid}\n")
        f.write(f"Number of training examples: {len(problem.train_pairs)}\n")
        f.write(f"Total turns: {len(all_responses)}\n")
        f.write("="*70 + "\n\n")
        
        f.write("COMPLETE CONVERSATION:\n")
        f.write("-"*70 + "\n")
        f.write(conversation)
        f.write("\n" + "="*70 + "\n\n")
        
        f.write("RESPONSE BREAKDOWN:\n")
        f.write("-"*70 + "\n")
        for turn_name, response in all_responses:
            f.write(f"\n{turn_name}:\n")
            f.write(response)
            f.write("\n" + "-"*50 + "\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("TARGET OUTPUT:\n")
        f.write(grid_to_string(problem.test_pairs[0].y))
    
    print(f"✅ Complete conversation saved to: {filename}")
    
    # Print complete conversation to terminal
    print("\n" + "="*70)
    print("📜 COMPLETE CONVERSATION TRANSCRIPT")
    print("="*70)
    print(conversation)
    
    print("\n" + "="*70)
    print("✅ Test completed!")
    print(f"Total conversation length: {len(conversation)} chars")
    print(f"Total turns: {len(all_responses)}")

if __name__ == "__main__":
    test_cot_multiturn()