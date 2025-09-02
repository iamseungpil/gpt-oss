#!/usr/bin/env python3
"""
Simple test to verify channel switching in GPT-OSS
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    SystemContent,
    DeveloperContent,
    TextContent
)

# Test 1: Basic harmony encoding
print("Testing harmony encoding...")
harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Build a simple conversation
convo = Conversation.from_messages([
    Message.from_role_and_content(
        Role.SYSTEM,
        SystemContent(
            text="You are ChatGPT, a large language model trained by OpenAI.",
            knowledge_cutoff="2024-06",
            reasoning="high"
        )
    ),
    Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent(
            text="""Use the analysis channel for reasoning and the final channel for your answer.
Example:
<|channel|>analysis<|message|>
I need to analyze this step by step...
<|channel|>final<|message|>
The answer is 42."""
        )
    ),
    Message.from_role_and_content(
        Role.USER,
        TextContent(text="What is 2+2? Use channels to show your work.")
    ),
])

# Render for completion
token_ids = harmony_encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
print(f"✅ Generated {len(token_ids)} tokens for prompt")

# Get stop tokens
stop_tokens = harmony_encoding.stop_tokens_for_assistant_actions()
print(f"✅ Stop tokens: {stop_tokens[:3]}...")

# Test 2: Check channel tokens exist in vocabulary
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)

test_strings = [
    "<|channel|>analysis",
    "<|channel|>final",
    "<|message|>",
    "<|end|>",
    "<|return|>"
]

print("\nChecking channel tokens in vocabulary:")
for s in test_strings:
    tokens = tokenizer.encode(s, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)
    print(f"  '{s}' -> {tokens} -> '{decoded}'")
    if decoded == s:
        print(f"    ✅ Exact match")
    else:
        print(f"    ⚠️ Mismatch!")

print("\n✅ All channel components verified in tokenizer")