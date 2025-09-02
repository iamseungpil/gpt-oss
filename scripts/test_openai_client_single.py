#!/usr/bin/env python3
"""
Single-turn Chat Completions test via OpenAI-compatible endpoint (e.g., vLLM/sglang).
Requires env vars: OPENAI_BASE_URL, OPENAI_API_KEY. Uses GPU handled by the server.
Verifies analysis/final channel usage and logs output.
"""

import os
import json
from pathlib import Path
from openai import OpenAI
from arc import train_problems
import numpy as np


def grid_to_string(grid: np.ndarray) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def main():
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    if not base_url:
        raise SystemExit("Set OPENAI_BASE_URL to your local server, e.g., http://localhost:8000/v1")

    client = OpenAI(base_url=base_url, api_key=api_key)

    prob = train_problems[0]
    examples = []
    for i, tr in enumerate(prob.train_pairs[:2], 1):
        examples.append(f"Example {i}:\nInput:\n{grid_to_string(tr.x)}\nOutput:\n{grid_to_string(tr.y)}")
    user_text = (
        "Solve this ARC puzzle. First analyze, then provide the solution grid.\n\n"
        + "\n\n".join(examples)
        + f"\n\nTest Input:\n{grid_to_string(prob.test_pairs[0].x)}\n"
    )

    messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI.", "reasoning": {"effort": "high"}},
        {"role": "developer", "content": (
            "# ARC Solver\n"
            "Use the analysis channel for reasoning, then the final channel for the grid.\n"
            "Grid format: digits 0-9, space-separated, one row per line. No extra text in final.\n\n"
            "Example format:\n"
            "<|channel|>analysis<|message|>\nExplain the rule briefly.\n"
            "<|channel|>final<|message|>\n1 2 3\n4 5 6\n7 8 9\n"
        )},
        {"role": "user", "content": user_text},
    ]

    resp = client.chat.completions.create(
        model=os.environ.get("MODEL", "openai/gpt-oss-20b"),
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )

    text = resp.choices[0].message.content or ""
    result = {
        "has_analysis": "<|channel|>analysis" in text,
        "has_final": "<|channel|>final" in text,
        "preview": text[:600],
    }
    out_dir = Path("logs"); out_dir.mkdir(exist_ok=True)
    (out_dir / "openai_client_single.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (out_dir / "openai_client_single.txt").write_text(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

