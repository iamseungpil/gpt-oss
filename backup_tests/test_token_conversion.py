#!/usr/bin/env python3
"""
Test token ID <-> text conversion to see if it preserves channel tokens
"""

from transformers import AutoTokenizer
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

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Create a simple conversation with channels
convo = Conversation.from_messages([
    Message.from_role_and_content(
        Role.SYSTEM,
        SystemContent(
            text="You are ChatGPT.",
            knowledge_cutoff="2024-06",
            reasoning="high"
        )
    ),
    Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent(
            text="Use the analysis channel for reasoning, then the final channel for your answer."
        )
    ),
    Message.from_role_and_content(
        Role.USER,
        TextContent(text="Test question")
    ),
])

# Get token IDs from harmony
token_ids = harmony_encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
print(f"1. Original token IDs from harmony: {len(token_ids)} tokens")
print(f"   First 20 tokens: {token_ids[:20]}")

# Decode to text (what training code does)
decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
print(f"\n2. Decoded text (first 500 chars):")
print(decoded_text[:500])

# Check if channel tokens are preserved
print(f"\n3. Channel tokens preserved in decoded text:")
print(f"   <|channel|>analysis: {'<|channel|>analysis' in decoded_text}")
print(f"   <|channel|>final: {'<|channel|>final' in decoded_text}")
print(f"   <|message|>: {'<|message|>' in decoded_text}")
print(f"   <|end|>: {'<|end|>' in decoded_text}")

# Re-encode the decoded text
re_encoded = tokenizer(decoded_text, return_tensors="pt").input_ids[0].tolist()
print(f"\n4. Re-encoded from decoded text: {len(re_encoded)} tokens")
print(f"   First 20 tokens: {re_encoded[:20]}")

# Compare
if token_ids == re_encoded:
    print("\n✅ PERFECT MATCH: Token IDs preserved through decode/encode cycle")
else:
    print(f"\n⚠️ MISMATCH: Original {len(token_ids)} tokens, re-encoded {len(re_encoded)} tokens")
    
    # Find differences
    min_len = min(len(token_ids), len(re_encoded))
    for i in range(min_len):
        if token_ids[i] != re_encoded[i]:
            print(f"   First difference at position {i}:")
            print(f"   Original: {token_ids[i]} -> '{tokenizer.decode([token_ids[i]])}'")
            print(f"   Re-encoded: {re_encoded[i]} -> '{tokenizer.decode([re_encoded[i]])}'")
            break