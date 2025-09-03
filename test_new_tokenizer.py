#!/usr/bin/env python3
"""
Test the new proper tokenizer usage for GPT-OSS
"""

import sys
sys.path.append('/home/ubuntu/gpt-oss')

from transformers import AutoModelForCausalLM, AutoTokenizer
from arc import validation_problems

def grid_to_string(grid):
    """Convert numpy grid to string format."""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

def create_harmony_prompt(problem):
    """Create proper harmony format messages for ARC problem."""
    # Build ARC examples (just first one for quick test)
    train_pair = problem.train_pairs[0]
    example_text = f"""Example 1:
Input:
{grid_to_string(train_pair.x)}
Output:
{grid_to_string(train_pair.y)}"""
    
    test_input = problem.test_pairs[0].x
    
    # Return messages in proper format
    messages = [
        {
            "role": "system", 
            "content": """You are ChatGPT, a large language model trained by OpenAI.

Reasoning: high

# Valid channels: analysis, commentary, final"""
        },
        {
            "role": "user", 
            "content": f"""# ARC Puzzle Solver

You are solving Abstract Reasoning Corpus (ARC) puzzles.

For each puzzle:
1. Use the analysis channel for examining patterns and reasoning
2. Identify the transformation rule from training examples
3. Apply the rule to the test input
4. Switch to the final channel for your solution grid

## Task
Solve this ARC puzzle:

{example_text}

Test Input:
{grid_to_string(test_input)}

What is the output grid?"""
        }
    ]
    
    return messages

def main():
    print("🧪 Testing new tokenizer usage...")
    
    # Load model and tokenizer
    model_id = "openai/gpt-oss-20b"
    print(f"📦 Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="cuda",
    )
    
    # Get test problem
    problem = validation_problems[0]
    print(f"📋 Test problem: {problem.uid}")
    
    # Create messages
    messages = create_harmony_prompt(problem)
    print(f"✅ Messages created: {len(messages)} items")
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    
    print(f"📝 Input tokens: {inputs['input_ids'].shape[1]}")
    
    # Generate short response for test
    print("🎯 Starting inference...")
    generated = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:])
    
    print("=" * 50)
    print("📄 RESPONSE PREVIEW:")
    print("=" * 50)
    print(response[:500] + "..." if len(response) > 500 else response)
    print("=" * 50)
    
    # Check for channels
    has_analysis = '<|channel|>analysis<|message|>' in response
    has_final = '<|channel|>final<|message|>' in response
    
    print(f"🔍 Analysis channel: {'✅' if has_analysis else '❌'}")
    print(f"🎯 Final channel: {'✅' if has_final else '❌'}")
    
    print("✅ Test completed!")

if __name__ == "__main__":
    main()