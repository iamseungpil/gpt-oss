#!/usr/bin/env python3
"""
Test Chat Template Consistency between Training and Inference
"""

import sys
sys.path.append('/home/ubuntu/gpt-oss')

from transformers import AutoTokenizer
from arc import train_problems, validation_problems

def grid_to_string(grid):
    """Convert numpy grid to string format."""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

def create_harmony_prompt_training(problem, tokenizer=None):
    """Training version - same as main_hf_trl_dapo_v2.py"""
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(
            f"Example {i}:\n"
            f"Input:\n{grid_to_string(train_pair.x)}\n"
            f"Output:\n{grid_to_string(train_pair.y)}"
        )
    
    test_input = problem.test_pairs[0].x
    examples_text = '\n\n'.join(examples)
    
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

You will naturally switch channels as you progress through the solution.

## Task
Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

What is the output grid?"""
        }
    ]
    
    return messages

def create_harmony_prompt_inference(problem):
    """Inference version - same as sequential_validation_v4.py"""
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(
            f"Example {i}:\n"
            f"Input:\n{grid_to_string(train_pair.x)}\n"
            f"Output:\n{grid_to_string(train_pair.y)}"
        )
    
    test_input = problem.test_pairs[0].x
    examples_text = '\n\n'.join(examples)
    
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

You will naturally switch channels as you progress through the solution.

## Task
Solve this ARC puzzle:

{examples_text}

Test Input:
{grid_to_string(test_input)}

What is the output grid?"""
        }
    ]
    
    return messages

def test_tokenizer_capabilities(tokenizer):
    """Test if tokenizer supports reasoning_effort"""
    test_msg = [{"role": "user", "content": "test"}]
    
    try:
        result = tokenizer.apply_chat_template(
            test_msg,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
            reasoning_effort="high"
        )
        print("✅ reasoning_effort is supported!")
        return True
    except TypeError as e:
        print(f"❌ reasoning_effort NOT supported: {e}")
        return False

def main():
    print("🧪 Testing Chat Template Consistency")
    print("=" * 60)
    
    # Load tokenizer
    model_id = "openai/gpt-oss-20b"
    print(f"📦 Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Test reasoning_effort support
    print("\n🔍 Testing reasoning_effort support:")
    supports_reasoning = test_tokenizer_capabilities(tokenizer)
    
    # Test with training problem 0
    print(f"\n📚 Testing with train_problems[0]: {train_problems[0].uid}")
    
    # Generate messages from both versions
    training_messages = create_harmony_prompt_training(train_problems[0])
    inference_messages = create_harmony_prompt_inference(train_problems[0])
    
    print(f"\n📊 Messages comparison:")
    print(f"Training messages length: {len(training_messages)}")
    print(f"Inference messages length: {len(inference_messages)}")
    print(f"Messages identical: {training_messages == inference_messages}")
    
    # Test tokenizer.apply_chat_template
    print(f"\n🔧 Testing tokenizer.apply_chat_template:")
    
    # Training style (string output)
    try:
        if supports_reasoning:
            training_prompt = tokenizer.apply_chat_template(
                training_messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="high"
            )
        else:
            training_prompt = tokenizer.apply_chat_template(
                training_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        print(f"✅ Training prompt generated: {len(training_prompt)} chars")
        print(f"Training prompt preview:\n{training_prompt[:300]}...")
    except Exception as e:
        print(f"❌ Training prompt failed: {e}")
    
    # Inference style (tensor output)
    try:
        if supports_reasoning:
            inference_inputs = tokenizer.apply_chat_template(
                inference_messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort="high"
            )
        else:
            inference_inputs = tokenizer.apply_chat_template(
                inference_messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )
        print(f"✅ Inference inputs generated: {inference_inputs['input_ids'].shape[1]} tokens")
        
        # Decode to compare
        decoded_inference = tokenizer.decode(inference_inputs['input_ids'][0])
        print(f"Inference prompt preview:\n{decoded_inference[:300]}...")
        
        # Compare prompts
        if 'training_prompt' in locals():
            print(f"\n🔍 Prompt comparison:")
            print(f"Training and inference prompts identical: {training_prompt.strip() == decoded_inference.strip()}")
    except Exception as e:
        print(f"❌ Inference inputs failed: {e}")
    
    print(f"\n📋 Token information:")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"EOS token: '{tokenizer.eos_token}'")
    
    print(f"\n✅ Chat template consistency test completed!")

if __name__ == "__main__":
    main()