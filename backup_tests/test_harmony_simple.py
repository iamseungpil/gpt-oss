#!/usr/bin/env python3
"""
Simple test for GPT-OSS channel output
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def test_simple_generation():
    """Test with a simple math problem to see channel usage"""
    
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
    
    # Initialize harmony encoding
    harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_tokens = harmony_encoding.stop_tokens_for_assistant_actions()
    
    print(f"✅ Model loaded. Stop tokens: {stop_tokens}")
    print("="*70)
    
    # Test cases
    test_cases = [
        {
            "name": "Simple math with reasoning",
            "user": "What is 25 * 17? Think step by step.",
            "developer": "You are a helpful assistant. Show your reasoning process.",
            "reasoning": ReasoningEffort.HIGH
        },
        {
            "name": "ARC-style pattern",
            "user": """Look at this pattern and complete it:
Input: 1 2 3
Output: 2 4 6

Input: 4 5 6
Output: 8 10 12

Input: 7 8 9
Output: ?

Think about the pattern first, then give your answer.""",
            "developer": "You are an expert at pattern recognition. Analyze the pattern before answering.",
            "reasoning": ReasoningEffort.HIGH
        },
    ]
    
    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"🧪 Test: {test['name']}")
        print(f"{'='*70}")
        
        # Build conversation
        messages = [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new().with_reasoning_effort(test['reasoning'])
            ),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions(test['developer'])
            ),
            Message.from_role_and_content(
                Role.USER,
                TextContent(text=test['user'])
            ),
        ]
        
        convo = Conversation.from_messages(messages)
        
        # Get input tokens
        input_ids = harmony_encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        
        # Print rendered input
        print("\n📥 Rendered input (first 200 chars):")
        input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(input_text[:200])
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids]).to(model.device)
        
        # Generate with different settings
        print("\n🔄 Generating response...")
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=512,  # Shorter for faster test
                temperature=0.7,
                top_p=0.9,
                eos_token_id=stop_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
            )
        
        # Decode response
        response = tokenizer.decode(output[0][input_tensor.shape[1]:], skip_special_tokens=False)
        
        print("\n📝 Full response:")
        print(response)
        
        # Analyze tokens
        print(f"\n📊 Analysis:")
        print(f"   - Contains <|channel|>: {'<|channel|>' in response}")
        print(f"   - Contains analysis: {'analysis' in response}")
        print(f"   - Contains final: {'final' in response}")
        print(f"   - Contains <|end|>: {'<|end|>' in response}")
        print(f"   - Contains <|return|>: {'<|return|>' in response}")
        
        # Save response
        filename = f"/tmp/gpt_oss_simple_{test['name'].replace(' ', '_')}.txt"
        with open(filename, 'w') as f:
            f.write(f"Test: {test['name']}\n")
            f.write(f"User: {test['user']}\n")
            f.write(f"{'='*70}\n")
            f.write("Full response:\n")
            f.write(response)
            f.write(f"\n{'='*70}\n")
            f.write("Input prompt:\n")
            f.write(input_text)
        print(f"💾 Saved to: {filename}")

if __name__ == "__main__":
    test_simple_generation()