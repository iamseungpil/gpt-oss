#!/usr/bin/env python3
"""
Test why final channel isn't reached with different prompt formats
"""

import os
import torch
import numpy as np
from arc import train_problems
from typing import Optional
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format"""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

def extract_final_channel(text: str) -> tuple[bool, str]:
    """Check if response reaches final channel and extract content"""
    has_final = "<|channel|>final<|" in text
    final_content = ""
    
    if has_final:
        parts = text.split("<|channel|>final<|")
        if len(parts) > 1:
            final_content = parts[1]
            if "message|>" in final_content:
                final_content = final_content.split("message|>", 1)[1]
            for marker in ["<|end|>", "<|channel|>", "<|return|>"]:
                if marker in final_content:
                    final_content = final_content.split(marker)[0]
            final_content = final_content.strip()
    
    return has_final, final_content

def test_prompts():
    """Test different prompt formats"""
    from unsloth import FastLanguageModel
    
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-BF16",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto"
    )
    
    # Test problem
    problem = train_problems[0]
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(f"Example {i}: Input:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}\n")
    examples_text = '\n'.join(examples)
    test_input = problem.test_pairs[0].x
    
    prompts = {
        "final_only": f"""<|start|>user<|message|>Solve this pattern:
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|channel|>final<|message|>""",
        
        "analysis_then_final": f"""<|start|>user<|message|>Solve this pattern:
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|channel|>analysis<|message|>Let me analyze the pattern step by step.<|end|>
<|start|>assistant<|channel|>final<|message|>""",
        
        "explicit_instruction": f"""<|start|>user<|message|>Solve this pattern. You MUST provide your answer in the final channel.
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|channel|>analysis<|message|>I'll analyze the pattern and provide the answer in the final channel.<|end|>
<|start|>assistant<|channel|>final<|message|>""",
        
        "no_analysis_start": f"""<|start|>user<|message|>Solve this pattern:
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|channel|>analysis<|message|>""",
        
        "analysis_with_continue": f"""<|start|>user<|message|>Solve this pattern:
{examples_text}
Test Input:
{grid_to_string(test_input)}
Test Output:<|end|>
<|start|>assistant<|channel|>analysis<|message|>Let me analyze briefly.<|end|>
<|start|>assistant<|channel|>final<|message|>The output grid is:<|end|>
<|start|>assistant<|message|>"""
    }
    
    results = {}
    
    for name, prompt in prompts.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Test with different max_new_tokens
        for max_tokens in [1024, 2048, 4096]:
            print(f"\nMax tokens: {max_tokens}")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            generated = response[len(prompt):]
            
            has_final, final_content = extract_final_channel(generated)
            
            print(f"  Has final channel: {has_final}")
            print(f"  Response length: {len(generated)}")
            
            if has_final:
                print(f"  Final content length: {len(final_content)}")
                print(f"  Final preview: {final_content[:200]}...")
            else:
                # Check if response was cut off
                if len(generated) >= max_tokens * 0.9:
                    print(f"  ⚠️ Response likely cut off (using {len(generated)} tokens)")
                # Show last part to see where it stopped
                print(f"  Last 200 chars: ...{generated[-200:]}")
            
            if name not in results:
                results[name] = {}
            results[name][max_tokens] = {
                "has_final": has_final,
                "response_length": len(generated),
                "final_length": len(final_content) if has_final else 0,
                "likely_cutoff": len(generated) >= max_tokens * 0.9
            }
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for name, token_results in results.items():
        print(f"\n{name}:")
        for max_tokens, data in token_results.items():
            status = "✅" if data["has_final"] else ("⚠️" if data["likely_cutoff"] else "❌")
            print(f"  {max_tokens} tokens: {status} Final: {data['has_final']}, Length: {data['response_length']}")
    
    # Save full results
    with open("/home/ubuntu/gpt_oss_arc_final/final_channel_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nFull results saved to final_channel_test_results.json")

if __name__ == "__main__":
    test_prompts()