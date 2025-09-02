#!/usr/bin/env python3
"""
Three-turn ARC interaction test on GPU 3 for problem 0.
Checks for analysis/final channels and basic gibberish signals.
Logs full outputs for inspection.
"""

import os
import re
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arc import train_problems


def grid_to_string(grid: np.ndarray) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def has_gibberish(text: str) -> bool:
    # Heuristic: excessive long repeated substrings or too many non-word symbols
    if len(text) == 0:
        return True
    # Repetition check
    tokens = text.split()
    if len(tokens) > 50:
        uniq = len(set(tokens)) / len(tokens)
        if uniq < 0.2:
            return True
    # Non-word density
    nonword = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in "|<>_/:-.,[]()\n"))
    if nonword / max(1, len(text)) > 0.2:
        return True
    return False


def main():
    # Force GPU 3 for this test
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    print("Loading tokenizer/model on GPU 3...")
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    problem = train_problems[0]
    print(f"ARC problem: {problem.uid}")

    # Build short examples
    examples = []
    for i, tr in enumerate(problem.train_pairs[:2], 1):
        examples.append(
            f"Example {i}:\nInput:\n{grid_to_string(tr.x)}\nOutput:\n{grid_to_string(tr.y)}"
        )
    test_input = problem.test_pairs[0].x
    examples_text = "\n\n".join(examples)

    # Turn 1: ask for analysis
    turn1 = (
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\n\nReasoning: high\n\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        "<|start|>developer<|message|># Instructions\n\n"
        "Use the analysis channel for reasoning and the final channel for answers.<|end|>"
        f"<|start|>user<|message|>I have an ARC puzzle. Here are the examples:\n\n{examples_text}\n\n"
        "Can you analyze what pattern or transformation rule is being applied here?<|end|><|start|>assistant"
    )

    ids1 = tok(turn1, return_tensors="pt").to(model.device)
    out1 = model.generate(
        **ids1,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=[200002],  # <|return|>
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
    )
    resp1 = tok.decode(out1[0][ids1.input_ids.shape[1]:], skip_special_tokens=False)

    # Turn 2: ask for solution (keep history)
    turn2 = (
        turn1 + resp1 +
        f"<|start|>user<|message|>Good analysis! Now here's the test input:\n\n{grid_to_string(test_input)}\n\n"
        "Provide only the grid as final answer.<|end|><|start|>assistant"
    )
    ids2 = tok(turn2, return_tensors="pt").to(model.device)
    out2 = model.generate(
        **ids2,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=[200002],
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
    )
    resp2 = tok.decode(out2[0][ids2.input_ids.shape[1]:], skip_special_tokens=False)

    # Turn 3: pre-inject final channel and request grid
    turn3 = (
        turn2 + resp2 +
        "<|start|>user<|message|>Please provide your final answer with just the output grid.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
    )
    ids3 = tok(turn3, return_tensors="pt").to(model.device)
    out3 = model.generate(
        **ids3,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=[200002],
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
    )
    resp3 = tok.decode(out3[0][ids3.input_ids.shape[1]:], skip_special_tokens=False)

    # Checks
    has_analysis = ("<|channel|>analysis" in resp1) or ("<|channel|>analysis" in resp2)
    has_final_2 = "<|channel|>final" in resp2
    has_final_3 = "<|channel|>final" in resp3

    # Basic gibberish flag
    gib_1 = has_gibberish(resp1)
    gib_2 = has_gibberish(resp2)
    gib_3 = has_gibberish(resp3)

    results = {
        "gpu": 3,
        "problem_uid": problem.uid,
        "turn1": {
            "has_analysis": has_analysis,
            "has_final": "<|channel|>final" in resp1,
            "gibberish": gib_1,
            "preview": resp1[:400],
        },
        "turn2": {
            "has_analysis": "<|channel|>analysis" in resp2,
            "has_final": has_final_2,
            "gibberish": gib_2,
            "preview": resp2[:500],
        },
        "turn3": {
            "has_final": has_final_3,
            "gibberish": gib_3,
            "preview": resp3[:500],
        },
    }

    # Save full logs
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    text_path = out_dir / f"gpu3_arc0_multiturn_{problem.uid}.txt"
    json_path = out_dir / f"gpu3_arc0_multiturn_{problem.uid}.json"

    with open(text_path, "w", encoding="utf-8") as f:
        f.write("TURN1 RESPONSE\n" + "="*40 + "\n" + resp1 + "\n\n")
        f.write("TURN2 RESPONSE\n" + "="*40 + "\n" + resp2 + "\n\n")
        f.write("TURN3 RESPONSE\n" + "="*40 + "\n" + resp3 + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(json.dumps(results, indent=2))
    print(f"\nSaved logs to {text_path} and {json_path}")


if __name__ == "__main__":
    main()
