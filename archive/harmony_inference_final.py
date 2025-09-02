#!/usr/bin/env python3
"""
GPT-OSS Harmony Format Inference - Final Consolidated Version
==============================================================
Based on official OpenAI Harmony documentation.

Key features:
1. Proper system message with reasoning levels (low/medium/high - "ultrathink")
2. Valid channels declaration: analysis, commentary, final
3. Natural channel switching without manual injection
4. Support for both simple and ARC puzzle tasks

Usage:
    # Simple test (no GPU needed - just shows prompt format)
    python harmony_inference_final.py demo
    
    # Run actual inference on GPU (specify GPU number)
    python harmony_inference_final.py run --gpu 3
    
    # Test with different reasoning levels
    python harmony_inference_final.py run --gpu 3 --reasoning high
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np
from typing import Optional, Dict, List, Tuple

# ARC problem import
try:
    from arc import train_problems, validation_problems
    HAS_ARC = True
except ImportError:
    HAS_ARC = False
    print("Warning: ARC module not found. Using mock data for demos.")


def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format."""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)


def create_harmony_prompt(task_type: str = "simple", reasoning_level: str = "high", problem=None) -> str:
    """
    Create a proper Harmony format prompt following official OpenAI documentation.
    
    Args:
        task_type: "simple" for math, "arc" for ARC puzzles
        reasoning_level: "low", "medium", or "high" (ultrathink)
        problem: ARC problem object (if task_type="arc")
    
    Returns:
        Properly formatted Harmony prompt string
    """
    # System message with reasoning level and valid channels
    system_msg = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-08-29

Reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final<|end|>"""
    
    if task_type == "simple":
        # Simple math problem
        developer_msg = """<|start|>developer<|message|># Problem Solver

When solving problems:
1. Use the analysis channel for your step-by-step reasoning
2. Use the final channel for your answer

You will naturally switch from analysis to final channel when ready.<|end|>"""
        
        user_msg = """<|start|>user<|message|>What is 2+2? Show your work.<|end|>"""
        
    elif task_type == "arc" and problem:
        # ARC puzzle
        developer_msg = """<|start|>developer<|message|># ARC Puzzle Solver

You are solving Abstract Reasoning Corpus (ARC) puzzles.

For each puzzle:
1. Use the analysis channel for examining patterns and reasoning
2. Identify the transformation rule from training examples
3. Apply the rule to the test input
4. Switch to the final channel for your solution grid

You will naturally switch channels as you progress through the solution.<|end|>"""
        
        # Build ARC examples
        examples = []
        for i, train_pair in enumerate(problem.train_pairs, 1):
            examples.append(
                f"Example {i}:\n"
                f"Input:\n{grid_to_string(train_pair.x)}\n"
                f"Output:\n{grid_to_string(train_pair.y)}"
            )
        
        test_input = problem.test_pairs[0].x
        examples_text = '\n\n'.join(examples)
        
        user_msg = f"""<|start|>user<|message|>Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

What is the output grid?<|end|>"""
    
    else:
        # Fallback demo
        developer_msg = """<|start|>developer<|message|>Use analysis channel for reasoning, final channel for answers.<|end|>"""
        user_msg = """<|start|>user<|message|>What is the capital of France?<|end|>"""
    
    # Combine all parts
    prompt = f"{system_msg}{developer_msg}{user_msg}<|start|>assistant"
    
    return prompt


def string_to_grid(grid_str: str) -> np.ndarray:
    """Convert string grid back to numpy array."""
    try:
        lines = grid_str.strip().split('\n')
        grid = []
        for line in lines:
            if line.strip():  # Skip empty lines
                row = [int(x) for x in line.split()]
                grid.append(row)
        return np.array(grid)
    except Exception as e:
        print(f"Error parsing grid: {e}")
        return None


def extract_final_grid(response: str) -> Optional[np.ndarray]:
    """Extract grid from final channel in response."""
    try:
        # Find final channel
        final_idx = response.find('<|channel|>final<|message|>')
        if final_idx == -1:
            return None
        
        # Extract content after final channel marker
        final_content = response[final_idx + len('<|channel|>final<|message|>'):]
        
        # Find the end marker
        end_idx = final_content.find('<|return|>')
        if end_idx == -1:
            end_idx = final_content.find('<|end|>')
        
        if end_idx != -1:
            final_content = final_content[:end_idx]
        
        # Parse the grid
        return string_to_grid(final_content)
    except Exception as e:
        print(f"Error extracting final grid: {e}")
        return None


def compare_grids(predicted: np.ndarray, target: np.ndarray) -> bool:
    """Compare two grids for exact match."""
    if predicted is None or target is None:
        return False
    if predicted.shape != target.shape:
        return False
    return np.array_equal(predicted, target)


def run_validation_batch(gpu: int = 5, reasoning_level: str = "high", num_problems: int = 10):
    """
    Run validation on multiple ARC problems.
    
    Args:
        gpu: GPU device number
        reasoning_level: "low", "medium", or "high"
        num_problems: Number of validation problems to test
    """
    if not HAS_ARC:
        print("ARC module not available!")
        return
    
    print(f"Starting ARC validation on {num_problems} problems...")
    print(f"GPU: {gpu}, Reasoning: {reasoning_level}")
    print("=" * 80)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model once
        print("Loading GPT-OSS model...")
        model_name = "openai/gpt-oss-20b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(device)
        
        results = []
        correct_count = 0
        total_start_time = time.time()
        
        # Process each problem
        for idx in range(num_problems):
            problem = validation_problems[idx]
            print(f"\n[{idx+1}/{num_problems}] Testing problem: {problem.uid}")
            print("-" * 40)
            
            problem_start_time = time.time()
            
            # Create prompt
            prompt = create_harmony_prompt("arc", reasoning_level, problem)
            
            # Tokenize and generate
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            
            print(f"Generating response...")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=30000,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=[tokenizer.eos_token_id, 200002],
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
            
            # Analyze response
            has_analysis = "<|channel|>analysis" in response
            has_final = "<|channel|>final" in response
            
            # Extract predicted grid
            predicted_grid = extract_final_grid(response)
            grid_extracted = predicted_grid is not None
            
            # Get target grid
            target_grid = problem.test_pairs[0].y
            
            # Compare
            is_correct = compare_grids(predicted_grid, target_grid)
            if is_correct:
                correct_count += 1
            
            problem_time = time.time() - problem_start_time
            
            # Store result
            result = {
                "problem_uid": problem.uid,
                "index": idx,
                "has_analysis": has_analysis,
                "has_final": has_final,
                "grid_extracted": grid_extracted,
                "correct": is_correct,
                "inference_time": round(problem_time, 2),
                "predicted_shape": predicted_grid.shape if predicted_grid is not None else None,
                "target_shape": target_grid.shape
            }
            results.append(result)
            
            # Print status
            print(f"✓ Analysis: {has_analysis}, Final: {has_final}")
            print(f"✓ Grid extracted: {grid_extracted}")
            print(f"✓ Shapes - Predicted: {result['predicted_shape']}, Target: {result['target_shape']}")
            print(f"✓ Correct: {'✅ YES' if is_correct else '❌ NO'}")
            print(f"✓ Time: {problem_time:.1f}s")
        
        total_time = time.time() - total_start_time
        accuracy_pct = (correct_count / num_problems) * 100
        
        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total problems: {num_problems}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {correct_count}/{num_problems} ({accuracy_pct:.1f}%)")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average time per problem: {total_time/num_problems:.1f}s")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/tmp/arc_validation_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "validation_results": results,
                "summary": {
                    "total_problems": num_problems,
                    "correct": correct_count,
                    "accuracy": f"{correct_count}/{num_problems} ({accuracy_pct:.1f}%)",
                    "total_time_seconds": round(total_time, 2),
                    "average_time_seconds": round(total_time/num_problems, 2)
                },
                "config": {
                    "gpu": gpu,
                    "reasoning_level": reasoning_level,
                    "model": "openai/gpt-oss-20b"
                }
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print detailed results table
        print("\nDetailed Results:")
        print("-" * 80)
        print(f"{'UID':<12} {'Analysis':<10} {'Final':<10} {'Grid':<10} {'Correct':<10} {'Time(s)':<10}")
        print("-" * 80)
        for r in results:
            print(f"{r['problem_uid']:<12} {str(r['has_analysis']):<10} {str(r['has_final']):<10} "
                  f"{str(r['grid_extracted']):<10} {'✅' if r['correct'] else '❌':<10} {r['inference_time']:<10.1f}")
        
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_prompt_format():
    """Show the proper Harmony format without running inference."""
    print("=" * 80)
    print("HARMONY FORMAT DEMONSTRATION")
    print("=" * 80)
    
    # Show simple example
    print("\n1. SIMPLE MATH PROBLEM (Reasoning: medium)")
    print("-" * 40)
    simple_prompt = create_harmony_prompt("simple", "medium")
    print(simple_prompt)
    
    print("\n" + "=" * 80)
    print("EXPECTED RESPONSE STRUCTURE:")
    print("-" * 40)
    print("""<|channel|>analysis<|message|>
Let me solve this step by step.
We have 2 + 2.
This is basic addition.
2 + 2 = 4
<|end|><|start|>assistant<|channel|>final<|message|>
4
<|return|>""")
    
    # Show ARC example if available
    if HAS_ARC:
        print("\n2. ARC PUZZLE (Reasoning: high - ultrathink)")
        print("-" * 40)
        problem = train_problems[0]
        arc_prompt = create_harmony_prompt("arc", "high", problem)
        print(arc_prompt[:1000] + "...")  # Show first 1000 chars
        
        print("\n" + "=" * 80)
        print("EXPECTED RESPONSE STRUCTURE:")
        print("-" * 40)
        print("""<|channel|>analysis<|message|>
Looking at the training examples...
[Detailed pattern analysis]
The transformation rule appears to be...
[Rule explanation]
Applying to test input...
<|end|><|start|>assistant<|channel|>final<|message|>
0 0 1 1 0
0 1 0 0 1
1 0 0 1 0
[rest of grid]
<|return|>""")
    
    print("\n" + "=" * 80)
    print("KEY POINTS:")
    print("-" * 40)
    print("✓ System message declares valid channels")
    print("✓ Reasoning level controls thinking depth (low/medium/high)")
    print("✓ Model naturally switches channels without injection")
    print("✓ Analysis channel for reasoning, final for answer")
    print("=" * 80)


def run_inference(gpu: int = 3, reasoning_level: str = "high", task: str = "arc"):
    """
    Run actual inference with GPT-OSS model.
    
    Args:
        gpu: GPU device number
        reasoning_level: "low", "medium", or "high"
        task: "simple" or "arc"
    """
    # Set GPU - Comment out as we'll set it externally
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    print(f"Loading GPT-OSS on GPU {gpu}...")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model
        model_name = "openai/gpt-oss-20b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(device)
        
        # Create prompt
        if task == "arc" and HAS_ARC:
            problem = train_problems[0]
            prompt = create_harmony_prompt("arc", reasoning_level, problem)
        else:
            prompt = create_harmony_prompt("simple", reasoning_level)
        
        # Tokenize and generate
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        print(f"Generating with reasoning={reasoning_level}...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=30000 if task == "arc" else 500,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, 200002],  # Include <|return|>
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
        
        # Analyze response
        print("\n" + "=" * 80)
        print("RESPONSE ANALYSIS:")
        print("-" * 40)
        
        has_analysis = "<|channel|>analysis" in response
        has_final = "<|channel|>final" in response
        
        print(f"✓ Analysis channel: {has_analysis}")
        print(f"✓ Final channel: {has_final}")
        print(f"✓ Response length: {len(response)} chars")
        
        if has_analysis and has_final:
            print("\n🎉 SUCCESS! Natural channel switching works!")
        elif has_analysis:
            print("\n⚠️ Partial: Only analysis channel found")
        else:
            print("\n❌ No proper channel usage")
        
        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/tmp/harmony_result_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "reasoning_level": reasoning_level,
                "task": task,
                "gpu": gpu,
                "response": response,
                "has_analysis": has_analysis,
                "has_final": has_final
            }, f, indent=2)
        
        print(f"\n💾 Saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure GPU has enough memory and conda env 'gptoss' is activated")


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS Harmony Format Tester")
    parser.add_argument("mode", choices=["demo", "run", "validate"], 
                        help="demo: show format, run: single inference, validate: batch validation")
    parser.add_argument("--gpu", type=int, default=3,
                        help="GPU number for inference (default: 3)")
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], default="high",
                        help="Reasoning level (default: high)")
    parser.add_argument("--task", choices=["simple", "arc"], default="arc",
                        help="Task type (default: arc)")
    parser.add_argument("--num_problems", type=int, default=10,
                        help="Number of validation problems to test (default: 10)")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demonstrate_prompt_format()
    elif args.mode == "validate":
        run_validation_batch(args.gpu, args.reasoning, args.num_problems)
    else:
        run_inference(args.gpu, args.reasoning, args.task)


if __name__ == "__main__":
    main()