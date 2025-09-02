#!/usr/bin/env python3
"""
Single-turn inference using the simpler harmony-like template (no openai_harmony),
replicating the known-good greedy config from test_inference.py, on GPU 3.
Checks whether analysis/final channels appear without forcing final injection.
"""

import os
import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def create_prompt(input_grid, expected_output_grid):
    return f"""<|system|>
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

<|user|>
Solve this ARC puzzle:

Input grid:
{json.dumps(input_grid)}

Expected output grid:
{json.dumps(expected_output_grid)}

What is the transformation rule and output?

<|assistant|>"""


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    # Simple 3x3 test
    input_grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    expected_output_grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    prompt = create_prompt(input_grid, expected_output_grid)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[200002],  # <|return|>
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )

    gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

    result = {
        "has_analysis": "<|channel|>analysis" in gen,
        "has_final": "<|channel|>final" in gen,
        "preview": gen[:600],
    }

    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "single_infer_gpu3.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (out_dir / "single_infer_gpu3.txt").write_text(gen)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
