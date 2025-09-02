#!/usr/bin/env python3
"""
Single-turn test (text prompt) on GPU 3.
No final-channel injection. Checks if model emits analysis and final channels.
"""

import os
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from arc import train_problems


def grid_to_string(grid: np.ndarray) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def build_prompt(problem) -> str:
    examples = []
    for i, tr in enumerate(problem.train_pairs[:2], 1):
        examples.append(f"Example {i}: Input:\n{grid_to_string(tr.x)}\nOutput:\n{grid_to_string(tr.y)}\n")
    examples_text = "\n".join(examples)

    test_input = problem.test_pairs[0].x

    prompt = f"""<|system|>
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

{examples_text}
Input grid:\n{grid_to_string(test_input)}

What is the transformation rule and output?

<|assistant|>"""
    return prompt


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    problem = train_problems[0]

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    prompt = build_prompt(problem)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=False,  # greedy for structure
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[200002],  # <|return|> token id used previously with success
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )

    gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

    result = {
        "gpu": 3,
        "problem_uid": problem.uid,
        "has_analysis": "<|channel|>analysis" in gen,
        "has_final": "<|channel|>final" in gen,
        "preview": gen[:600],
    }

    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"singleturn_text_gpu3_{problem.uid}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (out_dir / f"singleturn_text_gpu3_{problem.uid}.txt").write_text(gen)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
