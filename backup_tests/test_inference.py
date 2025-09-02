#!/usr/bin/env python3
"""Test inference with harmony prompt to diagnose generation issues"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_prompt():
    """Create a simple test prompt in harmony format"""
    
    # Simple ARC problem for testing
    test_problem = {
        "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "output": [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    }
    
    prompt = f"""<|system|>
You are an ARC puzzle solver. Analyze patterns and provide solutions.

<|developer|>
# ARC Puzzle Solver Instructions

You must solve ARC puzzles using TWO channels:

## Step 1: Analysis Channel
<|channel|>analysis<|message|>
- Study the input-output examples
- Identify the transformation rule
- Explain the pattern briefly

## Step 2: Final Channel  
<|channel|>final<|message|>
- Output ONLY the solution grid
- Numbers 0-9 separated by spaces
- One row per line
- NO explanations

## Example:
<|channel|>analysis<|message|>
The pattern rotates the input 90 degrees clockwise.
<|channel|>final<|message|>
1 2 3
4 5 6
7 8 9

<|user|>
Solve this ARC puzzle:

Input grid:
{json.dumps(test_problem['input'])}

Expected output grid:
{json.dumps(test_problem['output'])}

What is the transformation rule and output?

<|assistant|>"""
    
    return prompt

def test_generation_settings():
    """Test different generation settings to find what works"""
    
    logger.info("Loading model and tokenizer...")
    
    model_name = "openai/gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Force GPU 5
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # Maps to GPU 5
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    logger.info("Model loaded successfully")
    
    # Create test prompt
    prompt = create_test_prompt()
    logger.info(f"Prompt length: {len(prompt)} chars")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
    input_length = inputs.input_ids.shape[1]
    logger.info(f"Input tokens: {input_length}")
    
    # Test different generation configurations
    test_configs = [
        {
            "name": "Original (from training)",
            "params": {
                "max_new_tokens": 900,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
            }
        },
        {
            "name": "Lower temperature",
            "params": {
                "max_new_tokens": 900,
                "temperature": 0.3,
                "top_p": 0.9,
                "do_sample": True,
                "repetition_penalty": 1.5,
                "no_repeat_ngram_size": 3,
            }
        },
        {
            "name": "Greedy decoding",
            "params": {
                "max_new_tokens": 900,
                "do_sample": False,
                "repetition_penalty": 1.2,
            }
        },
        {
            "name": "High penalty",
            "params": {
                "max_new_tokens": 900,
                "temperature": 0.5,
                "top_p": 0.95,
                "do_sample": True,
                "repetition_penalty": 2.0,
                "no_repeat_ngram_size": 4,
            }
        },
    ]
    
    results = []
    
    for config in test_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config['name']}")
        logger.info(f"Parameters: {json.dumps(config['params'], indent=2)}")
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **config['params']
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            generated_only = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=False)
            
            # Check for repetition
            tokens = outputs[0][input_length:].tolist()
            unique_tokens = len(set(tokens))
            total_tokens = len(tokens)
            repetition_ratio = 1 - (unique_tokens / total_tokens) if total_tokens > 0 else 0
            
            # Check for channel usage
            has_analysis = "<|channel|>analysis" in generated_only
            has_final = "<|channel|>final" in generated_only
            
            # Look for grid pattern
            lines = generated_only.split('\n')
            grid_lines = [l for l in lines if any(c in '0123456789' for c in l)]
            
            result = {
                "config_name": config['name'],
                "tokens_generated": total_tokens,
                "unique_tokens": unique_tokens,
                "repetition_ratio": repetition_ratio,
                "has_analysis_channel": has_analysis,
                "has_final_channel": has_final,
                "grid_lines_found": len(grid_lines),
                "response_preview": generated_only[:500],
                "response_end": generated_only[-200:] if len(generated_only) > 200 else generated_only
            }
            
            results.append(result)
            
            logger.info(f"Generated {total_tokens} tokens")
            logger.info(f"Unique tokens: {unique_tokens} ({(1-repetition_ratio)*100:.1f}% unique)")
            logger.info(f"Channels: analysis={has_analysis}, final={has_final}")
            logger.info(f"Grid lines found: {len(grid_lines)}")
            logger.info(f"Response preview: {generated_only[:200]}...")
            
            # Check for severe repetition
            if repetition_ratio > 0.7:
                logger.warning(f"⚠️ HIGH REPETITION DETECTED: {repetition_ratio*100:.1f}% repetitive")
                # Show repeated pattern
                if total_tokens > 20:
                    logger.warning(f"First 20 tokens: {tokens[:20]}")
                    logger.warning(f"Last 20 tokens: {tokens[-20:]}")
            
        except Exception as e:
            logger.error(f"Error with config {config['name']}: {e}")
            results.append({
                "config_name": config['name'],
                "error": str(e)
            })
    
    # Save results
    output_file = Path("inference_test_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to {output_file}")
    
    # Summary
    logger.info("\n📊 Summary:")
    for result in results:
        if "error" not in result:
            logger.info(f"  {result['config_name']}:")
            logger.info(f"    - Repetition: {result['repetition_ratio']*100:.1f}%")
            logger.info(f"    - Channels: analysis={result['has_analysis_channel']}, final={result['has_final_channel']}")
            logger.info(f"    - Grid lines: {result['grid_lines_found']}")

if __name__ == "__main__":
    test_generation_settings()
