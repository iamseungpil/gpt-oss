#!/usr/bin/env python3
"""
Test forcing GPT-OSS to use final channel
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

def test_force_final_channel():
    """Test different methods to force final channel usage"""
    
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
    
    # Get first ARC problem
    problem = train_problems[0]
    
    # Build simple prompt
    examples = []
    for i, train_pair in enumerate(problem.train_pairs[:2], 1):  # Use only 2 examples for speed
        examples.append(f"Example {i}:\nInput:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}")
    
    test_input = problem.test_pairs[0].x
    examples_text = '\n\n'.join(examples)
    
    # Test different approaches
    test_cases = [
        {
            "name": "Baseline (chat template)",
            "method": "chat_template",
            "prompt": None
        },
        {
            "name": "Explicit final request",
            "method": "custom",
            "prompt": f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

You are an expert at solving ARC puzzles. Always provide your final answer in the final channel.<|end|><|start|>user<|message|>{examples_text}

Test Input:
{grid_to_string(test_input)}

Analyze the pattern and provide the output grid.<|end|><|start|>assistant"""
        },
        {
            "name": "Pre-inject final channel start",
            "method": "inject",
            "prompt": f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

You are an expert at solving ARC puzzles.<|end|><|start|>user<|message|>{examples_text}

Test Input:
{grid_to_string(test_input)}

Provide the output grid.<|end|><|start|>assistant<|channel|>analysis<|message|>Let me analyze the pattern: """,
            "continue_generation": True
        },
        {
            "name": "Multi-turn simulation",
            "method": "multiturn",
            "prompt": f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

You are an expert at solving ARC puzzles.<|end|><|start|>user<|message|>{examples_text}

Test Input:
{grid_to_string(test_input)}

What's the pattern?<|end|><|start|>assistant<|channel|>analysis<|message|>Looking at the examples, I can see that... [analyzing pattern]<|end|><|start|>assistant<|channel|>final<|message|>The output grid is:
""",
            "continue_from_final": True
        }
    ]
    
    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"🧪 Test: {test['name']}")
        print(f"{'='*70}")
        
        if test['method'] == 'chat_template':
            # Use standard chat template
            messages = [
                {"role": "system", "content": "You are an expert at solving ARC puzzles."},
                {"role": "user", "content": f"{examples_text}\n\nTest Input:\n{grid_to_string(test_input)}\n\nProvide the output grid."}
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = test['prompt']
        
        print(f"📥 Input ending: ...{input_text[-100:]}")
        
        # Tokenize
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        
        # Generate
        print("🔄 Generating...")
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
        
        print(f"\n📝 Response (first 400 chars):")
        print(response[:400])
        
        # Analyze
        print(f"\n📊 Analysis:")
        print(f"   - Has <|channel|>analysis: {'<|channel|>analysis' in response}")
        print(f"   - Has <|channel|>final: {'<|channel|>final' in response}")
        print(f"   - Has <|end|>: {'<|end|>' in response}")
        print(f"   - Response length: {len(response)} chars")
        
        # Extract final channel content if exists
        if '<|channel|>final<|message|>' in response:
            final_start = response.find('<|channel|>final<|message|>') + len('<|channel|>final<|message|>')
            final_end = response.find('<|', final_start)
            if final_end == -1:
                final_end = len(response)
            final_content = response[final_start:final_end].strip()
            print(f"\n✨ Final channel content:")
            print(final_content[:200])
        
        # Save
        filename = f"/tmp/force_final_{test['name'].replace(' ', '_')}.txt"
        with open(filename, 'w') as f:
            f.write(f"Test: {test['name']}\n")
            f.write(f"{'='*70}\n")
            f.write("Full response:\n")
            f.write(response)
        print(f"💾 Saved to: {filename}")
    
    print(f"\n{'='*70}")
    print("✅ All tests completed!")

if __name__ == "__main__":
    test_force_final_channel()