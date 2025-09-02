#!/usr/bin/env python3
"""
Test GPT-OSS inference with proper Harmony format for ARC tasks
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arc import train_problems
import numpy as np

# Import openai-harmony
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    SystemContent,
    DeveloperContent,
    TextContent,
    ReasoningEffort,
)

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def grid_to_string(grid: np.ndarray) -> str:
    """Convert numpy grid to string format"""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in grid)

def test_inference_variations():
    """Test different prompt variations to find what triggers channel usage"""
    
    print("🚀 Loading GPT-OSS model and tokenizer...")
    
    # Load model and tokenizer
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
    
    # Initialize harmony encoding
    harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_tokens = harmony_encoding.stop_tokens_for_assistant_actions()
    
    print(f"✅ Model loaded. Stop tokens: {stop_tokens}")
    print("="*70)
    
    # Get first ARC problem
    problem = train_problems[0]
    
    # Build examples
    examples = []
    for i, train_pair in enumerate(problem.train_pairs, 1):
        examples.append(f"Example {i}:\nInput:\n{grid_to_string(train_pair.x)}\nOutput:\n{grid_to_string(train_pair.y)}")
    
    test_input = problem.test_pairs[0].x
    examples_text = '\n\n'.join(examples)
    
    # Test different prompt variations
    variations = [
        {
            "name": "Basic",
            "user_content": f"""Analyze these input-output examples to identify the pattern, then apply it to solve the test input.

{examples_text}

Test Input:
{grid_to_string(test_input)}

Provide the output grid.""",
            "developer_instructions": "You are an expert at solving ARC puzzles.",
            "reasoning": ReasoningEffort.MEDIUM
        },
        {
            "name": "With channel instruction",
            "user_content": f"""Analyze these input-output examples to identify the pattern, then apply it to solve the test input.

{examples_text}

Test Input:
{grid_to_string(test_input)}

First analyze the pattern, then provide your final answer with the output grid.""",
            "developer_instructions": (
                "You are an expert at solving ARC puzzles. "
                "Think step by step about the pattern before providing your answer."
            ),
            "reasoning": ReasoningEffort.HIGH
        },
        {
            "name": "Explicit channel request",
            "user_content": f"""Analyze these input-output examples to identify the pattern, then apply it to solve the test input.

{examples_text}

Test Input:
{grid_to_string(test_input)}

Show your reasoning process first, then provide the final solution grid as numbers only.""",
            "developer_instructions": (
                "You are an expert pattern recognition AI specialized in solving Abstract Reasoning Corpus (ARC) puzzles. "
                "Always show your chain of thought reasoning before providing the final answer. "
                "The final answer should be a grid of numbers (0-9) with each row on a new line."
            ),
            "reasoning": ReasoningEffort.HIGH
        },
    ]
    
    for i, variant in enumerate(variations, 1):
        print(f"\n{'='*70}")
        print(f"🧪 Test {i}: {variant['name']}")
        print(f"{'='*70}")
        
        # Build conversation
        messages = [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new().with_reasoning_effort(variant['reasoning'])
            ),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions(variant['developer_instructions'])
            ),
            Message.from_role_and_content(
                Role.USER,
                TextContent(text=variant['user_content'])
            ),
        ]
        
        convo = Conversation.from_messages(messages)
        
        # Get token IDs
        input_ids = harmony_encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        input_ids = torch.tensor([input_ids]).to(model.device)
        
        # Generate
        print("Generating response...")
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=stop_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
            )
        
        # Decode response
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
        
        # Analyze response
        print("\n📝 Response preview (first 500 chars):")
        print(response[:500])
        
        # Check for channels
        has_analysis = "<|channel|>analysis" in response
        has_final = "<|channel|>final" in response
        has_commentary = "<|channel|>commentary" in response
        
        print(f"\n📊 Channel usage:")
        print(f"   - Analysis channel: {has_analysis}")
        print(f"   - Final channel: {has_final}")
        print(f"   - Commentary channel: {has_commentary}")
        
        # Check for special tokens
        print(f"\n🔍 Special tokens found:")
        special_tokens = ["<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|return|>", "<|call|>"]
        for token in special_tokens:
            if token in response:
                count = response.count(token)
                print(f"   - {token}: {count} times")
        
        # Save full response
        filename = f"/tmp/gpt_oss_response_{i}_{variant['name'].replace(' ', '_')}.txt"
        with open(filename, 'w') as f:
            f.write(f"Variant: {variant['name']}\n")
            f.write(f"Reasoning: {variant['reasoning']}\n")
            f.write(f"{'='*70}\n")
            f.write(response)
        print(f"\n💾 Full response saved to: {filename}")
    
    print(f"\n{'='*70}")
    print("✅ All tests completed!")
    print("Check the saved files for full responses.")

if __name__ == "__main__":
    test_inference_variations()