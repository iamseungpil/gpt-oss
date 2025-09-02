#!/usr/bin/env python3
"""
Sequential ARC Validation v2 - Optimized for 10 problems with single model load
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from typing import Optional

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

# ARC problem import
try:
    from arc import validation_problems
    HAS_ARC = True
except ImportError:
    HAS_ARC = False
    print("❌ ARC module not found.")

def log_with_timestamp(message):
    """Log message with timestamp and force flush."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format."""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

def string_to_grid(grid_str: str) -> np.ndarray:
    """Convert string grid back to numpy array."""
    try:
        lines = grid_str.strip().split('\n')
        grid = []
        for line in lines:
            if line.strip():
                row = [int(x) for x in line.split()]
                if row:
                    grid.append(row)
        return np.array(grid)
    except Exception as e:
        log_with_timestamp(f"❌ Error parsing grid: {e}")
        return None

def extract_final_grid(response: str) -> Optional[np.ndarray]:
    """Extract grid from final channel in response."""
    try:
        final_idx = response.find('<|channel|>final<|message|>')
        if final_idx == -1:
            return None
        
        final_content = response[final_idx + len('<|channel|>final<|message|>'):]
        end_idx = final_content.find('<|return|>')
        if end_idx == -1:
            end_idx = final_content.find('<|end|>')
        
        if end_idx != -1:
            final_content = final_content[:end_idx]
        
        return string_to_grid(final_content)
    except Exception as e:
        log_with_timestamp(f"❌ Error extracting grid: {e}")
        return None

def compare_grids(predicted: np.ndarray, target: np.ndarray) -> bool:
    """Compare two grids for exact match."""
    if predicted is None or target is None:
        return False
    if predicted.shape != target.shape:
        return False
    return np.array_equal(predicted, target)

def create_harmony_prompt(problem) -> str:
    """Create Harmony format prompt for ARC problem."""
    system_msg = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-09-01

Reasoning: high

# Valid channels: analysis, commentary, final<|end|>"""
    
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
    
    return f"{system_msg}{developer_msg}{user_msg}<|start|>assistant"

def process_single_problem(problem_idx: int, model, tokenizer, max_tokens: int = 15000):
    """Process a single ARC problem with pre-loaded model."""
    
    log_with_timestamp(f"=" * 60)
    log_with_timestamp(f"🔍 PROCESSING PROBLEM {problem_idx}")
    log_with_timestamp(f"=" * 60)
    
    problem = validation_problems[problem_idx]
    log_with_timestamp(f"📋 Problem UID: {problem.uid}")
    log_with_timestamp(f"📐 Target shape: {problem.test_pairs[0].y.shape}")
    log_with_timestamp(f"🎯 Max tokens: {max_tokens}")
    log_with_timestamp("-" * 60)
    
    try:
        # Create prompt and generate
        log_with_timestamp("🚀 Creating prompt and generating response...")
        start_inference = time.time()
        
        prompt = create_harmony_prompt(problem)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        log_with_timestamp(f"⚡ Starting inference with {input_ids.shape[1]} input tokens...")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, 200002],
            )
        
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
        inference_time = time.time() - start_inference
        
        log_with_timestamp(f"✅ Inference completed in {inference_time:.1f}s")
        log_with_timestamp(f"📝 Response length: {len(response):,} chars")
        
        # Analyze response
        has_analysis = "<|channel|>analysis" in response
        has_final = "<|channel|>final" in response
        
        log_with_timestamp(f"🔍 Analysis channel: {'✅' if has_analysis else '❌'}")
        log_with_timestamp(f"🎯 Final channel: {'✅' if has_final else '❌'}")
        
        # Extract and compare
        log_with_timestamp("🔧 Extracting predicted grid...")
        predicted_grid = extract_final_grid(response)
        target_grid = problem.test_pairs[0].y
        
        grid_extracted = predicted_grid is not None
        if grid_extracted:
            log_with_timestamp(f"📐 Predicted shape: {predicted_grid.shape}")
            log_with_timestamp(f"📐 Target shape: {target_grid.shape}")
            is_correct = compare_grids(predicted_grid, target_grid)
            log_with_timestamp(f"🎯 Shapes match: {'✅' if predicted_grid.shape == target_grid.shape else '❌'}")
            log_with_timestamp(f"🎯 Values match: {'✅' if is_correct else '❌'}")
        else:
            log_with_timestamp("❌ No grid extracted from response")
            is_correct = False
        
        # Result
        result = {
            "problem_idx": problem_idx,
            "problem_uid": problem.uid,
            "inference_time": round(inference_time, 2),
            "response_length": len(response),
            "has_analysis": has_analysis,
            "has_final": has_final,
            "grid_extracted": grid_extracted,
            "predicted_shape": predicted_grid.shape if predicted_grid is not None else None,
            "target_shape": target_grid.shape,
            "correct": is_correct,
            "max_tokens": max_tokens,
            "timestamp": datetime.now().isoformat()
        }
        
        log_with_timestamp(f"{'🎉 SUCCESS' if is_correct else '❌ FAILED'}: Problem {problem_idx} completed")
        return result, response
        
    except Exception as e:
        log_with_timestamp(f"❌ Error processing problem {problem_idx}: {e}")
        import traceback
        traceback.print_exc()
        
        result = {
            "problem_idx": problem_idx,
            "problem_uid": problem.uid if 'problem' in locals() else "unknown",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return result, None

def run_sequential_validation_optimized(start_idx: int = 0, num_problems: int = 10, gpu: int = 5, max_tokens: int = 15000):
    """Run validation problems sequentially with single model load."""
    
    if not HAS_ARC:
        log_with_timestamp("❌ ARC module not available!")
        return
    
    log_with_timestamp("🚀 STARTING SEQUENTIAL ARC VALIDATION")
    log_with_timestamp(f"📊 Problems: {start_idx} to {start_idx + num_problems - 1}")
    log_with_timestamp(f"🖥️ GPU: {gpu}, Max tokens: {max_tokens:,}")
    log_with_timestamp("=" * 80)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model once
        log_with_timestamp("📦 Loading GPT-OSS model (one-time setup)...")
        start_load = time.time()
        
        model_name = "openai/gpt-oss-20b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        
        log_with_timestamp(f"⚡ Loading model to {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(device)
        
        load_time = time.time() - start_load
        log_with_timestamp(f"✅ Model loaded successfully in {load_time:.1f}s")
        log_with_timestamp("=" * 80)
        
        results = []
        responses = {}
        correct_count = 0
        total_start = time.time()
        
        # Process all problems
        for i in range(start_idx, start_idx + num_problems):
            problem_num = i - start_idx + 1
            log_with_timestamp(f"\n🔄 [{problem_num}/{num_problems}] Starting problem {i}...")
            
            result, response = process_single_problem(i, model, tokenizer, max_tokens)
            results.append(result)
            
            if response:
                responses[str(i)] = response
            
            if result.get('correct', False):
                correct_count += 1
            
            # Save intermediate progress
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            progress_file = f"/tmp/arc_seq_progress_{timestamp}.json"
            
            progress_data = {
                "completed_problems": problem_num,
                "total_problems": num_problems,
                "correct_so_far": correct_count,
                "accuracy_so_far": f"{correct_count}/{problem_num} ({100*correct_count/problem_num:.1f}%)",
                "results": results,
                "config": {
                    "start_idx": start_idx,
                    "num_problems": num_problems,
                    "gpu": gpu,
                    "max_tokens": max_tokens
                }
            }
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            log_with_timestamp(f"💾 Progress saved: {progress_file}")
            log_with_timestamp(f"📊 Running accuracy: {correct_count}/{problem_num} ({100*correct_count/problem_num:.1f}%)")
            
            if i < start_idx + num_problems - 1:
                log_with_timestamp("⏱️ Brief pause before next problem...")
                time.sleep(2)
        
        # Final results
        total_time = time.time() - total_start
        accuracy_pct = (correct_count / num_problems) * 100
        
        log_with_timestamp("\n" + "=" * 80)
        log_with_timestamp("🎉 SEQUENTIAL VALIDATION COMPLETE!")
        log_with_timestamp("=" * 80)
        log_with_timestamp(f"📊 Total problems: {num_problems}")
        log_with_timestamp(f"✅ Correct: {correct_count}")
        log_with_timestamp(f"🎯 Final accuracy: {correct_count}/{num_problems} ({accuracy_pct:.1f}%)")
        log_with_timestamp(f"⏱️ Total time: {total_time/60:.1f} minutes")
        log_with_timestamp(f"⚡ Average per problem: {total_time/num_problems:.1f}s")
        
        # Save final results
        final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = f"/tmp/arc_sequential_final_{final_timestamp}.json"
        
        final_data = {
            "validation_results": results,
            "summary": {
                "total_problems": num_problems,
                "correct": correct_count,
                "accuracy": f"{correct_count}/{num_problems} ({accuracy_pct:.1f}%)",
                "total_time_seconds": round(total_time, 2),
                "average_time_seconds": round(total_time/num_problems, 2),
                "model_load_time": round(load_time, 2)
            },
            "config": {
                "start_idx": start_idx,
                "num_problems": num_problems,
                "gpu": gpu,
                "max_tokens": max_tokens,
                "model": "openai/gpt-oss-20b"
            }
        }
        
        with open(final_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        log_with_timestamp(f"💾 Final results saved: {final_file}")
        
        # Results table
        log_with_timestamp("\n📋 DETAILED RESULTS:")
        log_with_timestamp("-" * 80)
        log_with_timestamp(f"{'Idx':<4} {'UID':<12} {'Analysis':<8} {'Final':<8} {'Grid':<8} {'Correct':<8} {'Time(s)':<8}")
        log_with_timestamp("-" * 80)
        
        for r in results:
            if 'error' not in r:
                status = "✅" if r['correct'] else "❌"
                log_with_timestamp(f"{r['problem_idx']:<4} {r['problem_uid']:<12} {str(r['has_analysis']):<8} "
                      f"{str(r['has_final']):<8} {str(r['grid_extracted']):<8} "
                      f"{status:<8} {r.get('inference_time', 0):<8.1f}")
        
        log_with_timestamp("=" * 80)
        log_with_timestamp("🏁 VALIDATION COMPLETE!")
        
    except Exception as e:
        log_with_timestamp(f"❌ Fatal error during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sequential ARC Validation v2")
    parser.add_argument("--start", type=int, default=0, help="Starting problem index")
    parser.add_argument("--num", type=int, default=10, help="Number of problems")
    parser.add_argument("--gpu", type=int, default=5, help="GPU number")
    parser.add_argument("--tokens", type=int, default=15000, help="Max tokens")
    
    args = parser.parse_args()
    
    # Import torch here to avoid early import
    import torch
    
    run_sequential_validation_optimized(args.start, args.num, args.gpu, args.tokens)