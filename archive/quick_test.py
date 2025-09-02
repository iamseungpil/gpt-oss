#!/usr/bin/env python3
"""Quick test without full model loading"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)

# Test prompt with channels
prompt = """<|start|>system<|message|>You are ChatGPT.
Reasoning: high<|end|><|start|>developer<|message|>Always use channels:
<|channel|>analysis<|message|>
For your reasoning
<|channel|>final<|message|>
For your final answer<|end|><|start|>user<|message|>What is 2+2?<|end|><|start|>assistant<|channel|>"""

print("PROMPT ANALYSIS:")
print("="*50)
print(f"Prompt length: {len(prompt)} chars")
print(f"Has <|channel|>: {'<|channel|>' in prompt}")
print(f"Has developer message: {'developer' in prompt}")

# Tokenize
tokens = tokenizer.encode(prompt, add_special_tokens=False)
print(f"\nTokenized: {len(tokens)} tokens")

# Decode back
decoded = tokenizer.decode(tokens, skip_special_tokens=False)
print(f"\nRound-trip successful: {decoded == prompt}")

# Check specific tokens
channel_tokens = tokenizer.encode("<|channel|>", add_special_tokens=False)
analysis_tokens = tokenizer.encode("analysis", add_special_tokens=False)
final_tokens = tokenizer.encode("final", add_special_tokens=False)
message_tokens = tokenizer.encode("<|message|>", add_special_tokens=False)

print("\nSpecial tokens:")
print(f"<|channel|> -> {channel_tokens}")
print(f"analysis -> {analysis_tokens}")
print(f"final -> {final_tokens}")
print(f"<|message|> -> {message_tokens}")

# Check if prompt ends correctly for generation
if prompt.endswith("<|channel|>"):
    print("\n✅ Prompt ends with channel token - ready for channel generation!")
elif prompt.endswith("assistant"):
    print("\n⚠️ Prompt ends with 'assistant' - may not generate channel")

print("\nExpected model continuation:")
print("  Should generate: 'analysis<|message|>' or similar")
print("  Then reasoning content")
print("  Then '<|channel|>final<|message|>'")
print("  Then final answer")