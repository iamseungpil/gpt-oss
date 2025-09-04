#!/usr/bin/env python3
"""
Single GPU Medium Reasoning Test - 12k tokens, reasoning=medium
Testing channel switching with proper harmony format
"""

import os
import sys
import json
import time
import torch
from datetime import datetime

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

from transformers import AutoTokenizer, AutoModelForCausalLM
from arc import train_problems

def log_with_timestamp(message):
    """Enhanced logging with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def create_harmony_prompt(problem, tokenizer):
    """Create harmony-compliant prompt with reasoning=medium"""
    # Build training examples
    train_examples_str = []
    for i, train_pair in enumerate(problem.train_pairs):
        input_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in train_pair.x)
        output_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in train_pair.y)
        train_examples_str.append(f"Example {i+1}:\nInput:\n{input_str}\n\nOutput:\n{output_str}")
    
    test_input = problem.test_pairs[0].x
    test_input_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in test_input)
    
    user_content = f"""# ARC Puzzle Solver

You are an expert at solving Abstract Reasoning Corpus (ARC) puzzles. Analyze the pattern and solve the test case.

## Training Examples:
{chr(10).join(train_examples_str)}

## Test Input:
{test_input_str}

Please find the pattern and provide the solution grid."""
    
    # GPT-OSS harmony format with reasoning=medium
    messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. You excel at pattern recognition and logical reasoning. Reasoning: medium"},
        {"role": "user", "content": user_content}
    ]
    
    # Apply chat template with reasoning_effort
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="medium"
        )
        log_with_timestamp("✅ Applied chat template with reasoning_effort=medium")
    except Exception as e:
        log_with_timestamp(f"⚠️ Fallback to basic chat template: {e}")
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    # Add channel start token for GPT-OSS
    prompt = prompt + "<|channel|>"
    return prompt

def run_single_gpu_medium_test():
    """Run single GPU test with reasoning=medium, 12k tokens"""
    log_with_timestamp("🚀 Starting Single GPU Medium Test - 12k tokens")
    log_with_timestamp("🎯 Target: reasoning=medium with proper channel switching")
    
    # Load model on GPU 6
    log_with_timestamp("📦 Loading GPT-OSS model on GPU 6...")
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "openai/gpt-oss-20b",
        trust_remote_code=True,
        padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log_with_timestamp("✅ Model loaded on GPU 6")
    
    # Test problems 0-9
    results = []
    total_start_time = time.time()
    
    for problem_idx in range(10):
        log_with_timestamp(f"🔄 Processing problem {problem_idx}...")
        
        problem = train_problems[problem_idx]
        prompt = create_harmony_prompt(problem, tokenizer)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_tokens = inputs["input_ids"].shape[1]
        log_with_timestamp(f"📝 Problem {problem_idx}: {input_tokens} input tokens")
        
        # Generate with 12k tokens
        start_time = time.time()
        
        with torch.no_grad():
            torch.cuda.empty_cache()  # Clean memory
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=12000,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id, 200002],  # Include channel end token
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
            
            torch.cuda.empty_cache()  # Clean memory after generation
        
        generation_time = time.time() - start_time
        result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        tokens_generated = len(outputs[0]) - inputs["input_ids"].shape[1]
        tokens_per_second = tokens_generated / generation_time
        
        # Analyze channel switching
        analysis_found = "<|channel|>analysis<|message|>" in result
        final_found = "<|channel|>final<|message|>" in result
        
        log_with_timestamp(f"✅ Problem {problem_idx} completed!")
        log_with_timestamp(f"   📊 Generated: {tokens_generated} tokens in {generation_time:.1f}s")
        log_with_timestamp(f"   ⚡ Speed: {tokens_per_second:.2f} tokens/second")
        log_with_timestamp(f"   🔍 Analysis channel: {analysis_found}")
        log_with_timestamp(f"   🔍 Final channel: {final_found}")
        
        # Save individual result
        problem_result = {
            'problem_idx': problem_idx,
            'input_tokens': input_tokens,
            'tokens_generated': tokens_generated,
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
            'analysis_channel_found': analysis_found,
            'final_channel_found': final_found,
            'result': result
        }
        results.append(problem_result)
        
        # Log memory usage
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        log_with_timestamp(f"   🧠 GPU Memory: {memory_allocated:.1f}GB")
    
    total_time = time.time() - total_start_time
    
    # Calculate final statistics
    total_tokens = sum(r['tokens_generated'] for r in results)
    avg_speed = sum(r['tokens_per_second'] for r in results) / len(results)
    analysis_success = sum(1 for r in results if r['analysis_channel_found'])
    final_success = sum(1 for r in results if r['final_channel_found'])
    
    log_with_timestamp("📊 FINAL RESULTS - MEDIUM REASONING 12K TEST:")
    log_with_timestamp(f"   🎯 Problems processed: {len(results)}")
    log_with_timestamp(f"   📝 Total tokens generated: {total_tokens:,}")
    log_with_timestamp(f"   ⏱️ Total time: {total_time:.1f}s")
    log_with_timestamp(f"   ⚡ Average speed: {avg_speed:.2f} tokens/second")
    log_with_timestamp(f"   🔍 Analysis channel success: {analysis_success}/10 ({analysis_success*10}%)")
    log_with_timestamp(f"   🔍 Final channel success: {final_success}/10 ({final_success*10}%)")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"medium_12k_test_results_{timestamp}.json"
    
    summary = {
        "timestamp": timestamp,
        "test_type": "single_gpu_medium_12k",
        "reasoning_effort": "medium",
        "max_tokens": 12000,
        "num_problems": 10,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_tokens_per_second": avg_speed,
        "analysis_channel_success_rate": analysis_success / 10,
        "final_channel_success_rate": final_success / 10,
        "results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"💾 Results saved to {output_file}")
    log_with_timestamp("🎉 Medium reasoning 12k test completed successfully!")
    
    return summary

if __name__ == "__main__":
    try:
        run_single_gpu_medium_test()
    except Exception as e:
        log_with_timestamp(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)