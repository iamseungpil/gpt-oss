#!/usr/bin/env python3
"""
Test GPT-OSS with proper Harmony format based on official OpenAI documentation.
Key features:
1. Proper system message with reasoning levels (low/medium/high)
2. Valid channels declaration: analysis, commentary, final
3. Natural channel switching without manual injection
4. Large max_new_tokens for complete responses
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from arc import train_problems
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format."""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)


def create_harmony_prompt(problem, reasoning_level="high"):
    """
    Create a proper Harmony format prompt following official OpenAI documentation.
    
    Args:
        problem: ARC problem object with train_pairs and test_pairs
        reasoning_level: "low", "medium", or "high" (ultrathink)
    """
    # Build training examples
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(
            f"Example {i}:\n"
            f"Input:\n{grid_to_string(train_pair.x)}\n"
            f"Output:\n{grid_to_string(train_pair.y)}"
        )
    
    test_input = problem.test_pairs[0].x
    examples_text = '\n\n'.join(examples)
    
    # Build proper Harmony prompt with correct system message
    # Based on official OpenAI Harmony documentation
    prompt = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-08-29

Reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final<|end|><|start|>developer<|message|># ARC Puzzle Solver

You are solving Abstract Reasoning Corpus (ARC) puzzles. 

For each puzzle:
1. Use the analysis channel for your step-by-step reasoning
2. Examine the training examples to identify patterns
3. Apply the discovered pattern to the test input
4. Switch to the final channel for your solution grid

You will naturally switch from analysis to final channel when ready.<|end|><|start|>user<|message|>Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

What is the output grid?<|end|><|start|>assistant"""
    
    return prompt


def extract_grid_from_response(response: str) -> Optional[str]:
    """Extract grid from the final channel response."""
    try:
        if "<|channel|>final<|message|>" in response:
            final_start = response.find("<|channel|>final<|message|>")
            final_content = response[final_start + len("<|channel|>final<|message|>"):]
            
            # Find end of final channel
            for end_token in ["<|return|>", "<|end|>", "<|channel|>"]:
                if end_token in final_content:
                    final_content = final_content.split(end_token)[0]
                    break
            
            return final_content.strip()
    except Exception as e:
        print(f"Grid extraction error: {e}")
    
    return None


def test_inference(reasoning_level="high", max_tokens=4000):
    """
    Test GPT-OSS inference with proper Harmony format.
    
    Args:
        reasoning_level: "low", "medium", or "high" (ultrathink)
        max_tokens: Maximum new tokens to generate
    """
    print("=" * 80)
    print(f"Testing Harmony Format (Reasoning: {reasoning_level})")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\n1. Loading GPT-OSS model on GPU 5...")
    model_name = "openai/gpt-oss-20b"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ Model loaded successfully")
    
    # Get first ARC problem
    print("\n2. Loading ARC problem...")
    problem = train_problems[0]
    print(f"✓ Problem loaded with {len(problem.train_pairs)} training examples")
    
    # Create prompt with proper Harmony format
    print("\n3. Creating Harmony format prompt...")
    prompt = create_harmony_prompt(problem, reasoning_level)
    
    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    print(f"✓ Input length: {input_ids.shape[1]} tokens")
    
    # Generate with proper settings
    print(f"\n4. Generating response (max_new_tokens: {max_tokens})...")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id, 200002],  # Include <|return|> token
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
    
    print("\n5. Analyzing response structure...")
    print("-" * 40)
    
    # Check for channels
    analysis = {
        "has_analysis_channel": "<|channel|>analysis" in response,
        "has_final_channel": "<|channel|>final" in response,
        "has_commentary_channel": "<|channel|>commentary" in response,
        "starts_with_channel": response.startswith("<|channel|>"),
        "response_length": len(response),
        "reasoning_level": reasoning_level
    }
    
    print(f"✓ Analysis channel: {analysis['has_analysis_channel']}")
    print(f"✓ Final channel: {analysis['has_final_channel']}")
    print(f"✓ Commentary channel: {analysis['has_commentary_channel']}")
    print(f"✓ Response length: {analysis['response_length']} chars")
    
    # Extract channel contents
    if analysis['has_analysis_channel']:
        analysis_start = response.find("<|channel|>analysis<|message|>")
        if analysis_start != -1:
            analysis_content = response[analysis_start + len("<|channel|>analysis<|message|>"):]
            next_channel = analysis_content.find("<|channel|>")
            if next_channel != -1:
                analysis_content = analysis_content[:next_channel]
            elif "<|end|>" in analysis_content:
                analysis_content = analysis_content.split("<|end|>")[0]
            print(f"\n📊 Analysis channel (first 300 chars):")
            print(f"  {analysis_content[:300]}...")
    
    if analysis['has_final_channel']:
        final_content = extract_grid_from_response(response)
        if final_content:
            print(f"\n✅ Final channel content:")
            print(f"  {final_content[:500]}")
            
            # Check if it looks like a grid
            lines = final_content.strip().split('\\n')
            is_grid = all(
                all(c in '0123456789 ' for c in line) 
                for line in lines if line.strip()
            )
            if is_grid and len(lines) > 0:
                print("  ✓ Valid grid format detected!")
            else:
                print("  ⚠️ Output doesn't appear to be a valid grid")
    else:
        print(f"\n⚠️ No final channel found!")
        print(f"Response ends with: ...{response[-200:]}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/tmp/harmony_{reasoning_level}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "reasoning_level": reasoning_level,
            "max_tokens": max_tokens,
            "prompt": prompt[:1000],  # First 1000 chars
            "response": response,
            "analysis": analysis,
            "timestamp": timestamp
        }, f, indent=2)
    
    print(f"\n💾 Full results saved to: {output_file}")
    
    # Success check
    if analysis['has_analysis_channel'] and analysis['has_final_channel']:
        print("\n🎉 SUCCESS! Natural channel switching works!")
        return True
    elif analysis['has_analysis_channel']:
        print("\n⚠️ Partial: Only analysis channel, no final channel")
        return False
    else:
        print("\n❌ FAIL: No proper channel usage")
        return False


def test_simple():
    """Test with a simple math problem to verify channel switching."""
    print("\n" + "=" * 80)
    print("Testing Simple Problem (2+2)")
    print("=" * 80)
    
    # Simple prompt with proper Harmony format
    prompt = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-08-29

Reasoning: medium

# Valid channels: analysis, commentary, final<|end|><|start|>developer<|message|>When solving problems:
1. Use the analysis channel for your step-by-step reasoning
2. Use the final channel for your answer<|end|><|start|>user<|message|>What is 2+2? Show your work.<|end|><|start|>assistant"""
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Generate
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id, 200002],
        )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
    
    print(f"\nResponse (first 500 chars):")
    print(response[:500])
    
    has_analysis = "<|channel|>analysis" in response
    has_final = "<|channel|>final" in response
    
    print(f"\n✓ Analysis channel: {has_analysis}")
    print(f"✓ Final channel: {has_final}")
    
    return has_analysis and has_final


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    test_type = sys.argv[1] if len(sys.argv) > 1 else "arc"
    reasoning = sys.argv[2] if len(sys.argv) > 2 else "high"
    
    if test_type == "simple":
        # Test simple math problem
        success = test_simple()
        if success:
            print("\n✅ Simple test PASSED! Channel switching works.")
        else:
            print("\n❌ Simple test FAILED. Check model/prompt.")
    
    elif test_type == "arc":
        # Test ARC problem with specified reasoning level
        success = test_inference(reasoning_level=reasoning, max_tokens=5000)
        if success:
            print("\n✅ ARC test PASSED! Ready for training.")
        else:
            print("\n⚠️ ARC test incomplete. May need larger max_tokens.")
    
    else:
        print("Usage: python test_harmony_proper.py [simple|arc] [low|medium|high]")
        print("  simple - Test with 2+2 math problem")
        print("  arc - Test with ARC puzzle (default)")
        print("  Reasoning levels: low, medium, high (ultrathink)")
        sys.exit(1)