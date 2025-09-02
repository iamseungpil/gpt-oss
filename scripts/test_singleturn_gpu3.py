#!/usr/bin/env python3
"""
Single-turn ARC test on GPU 3 using Harmony encoding.
Verifies whether the model emits analysis and final channels without injecting final.
Logs full response and extracts a grid from the final channel if present.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from arc import train_problems

from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    SystemContent,
    DeveloperContent,
    TextContent,
)


def grid_to_string(grid: np.ndarray) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def parse_grid_block(text: str, target_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    lines = text.strip().split("\n")
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(h in line.lower() for h in ["grid:", "output:", "answer:", "result:", "example", "input:"]):
            continue
        parts = line.replace(",", " ").split()
        vals = []
        for p in parts:
            try:
                if "." in p:
                    n = int(float(p))
                else:
                    n = int(p)
            except ValueError:
                vals = []
                break
            if 0 <= n <= 9:
                vals.append(n)
            else:
                vals = []
                break
        if vals:
            rows.append(vals)
        elif rows:
            break
    if rows:
        if len(set(len(r) for r in rows)) == 1:
            arr = np.array(rows)
            if target_shape is None or arr.shape == target_shape:
                return arr
    return None


def parse_grid_from_response(response: str, target_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    # Prefer final channel content
    m = re.search(r"<\|channel\|>final<\|message\|>([\s\S]*?)(?:<\|end\|>|$)", response)
    if m:
        block = m.group(1).strip()
        grid = parse_grid_block(block, target_shape)
        if grid is not None:
            return grid
    # Fallback: look into code blocks
    for block in re.findall(r"```(?:python|text)?\n?([\s\S]*?)```", response):
        grid = parse_grid_block(block.strip(), target_shape)
        if grid is not None:
            return grid
    # Last resort: whole response
    return parse_grid_block(response, target_shape)


class StopOnSequences(StoppingCriteria):
    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        self.stop_sequences = [s for s in stop_sequences if s]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore[override]
        if not self.stop_sequences:
            return False
        gen = input_ids[0].tolist()
        for seq in self.stop_sequences:
            L = len(seq)
            if L <= len(gen) and gen[-L:] == seq:
                return True
        return False


def main():
    # Use GPU 3
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # Load tokenizer/model
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )

    # Harmony encoding
    harmony = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_tokens = harmony.stop_tokens_for_assistant_actions()

    # Problem 0
    prob = train_problems[0]
    examples = []
    for i, tr in enumerate(prob.train_pairs[:2], 1):
        examples.append(f"Example {i}:\nInput:\n{grid_to_string(tr.x)}\nOutput:\n{grid_to_string(tr.y)}")
    test_in = prob.test_pairs[0].x

    user_text = (
        "Solve this ARC puzzle. First analyze, then provide the solution grid.\n\n"
        + "\n\n".join(examples)
        + f"\n\nTest Input:\n{grid_to_string(test_in)}\n"
    )

    convo = Conversation.from_messages([
        Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent(text="You are ChatGPT, a large language model trained by OpenAI.", knowledge_cutoff="2024-06", reasoning="high"),
        ),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent(text=(
                "# ARC Solver\n"
                "Use the analysis channel for reasoning, then the final channel for the grid.\n"
                "Grid format: digits 0-9, space-separated, one row per line. No extra text in final.\n\n"
                "Example format:\n"
                "<|channel|>analysis<|message|>\n"
                "Identify the rule from examples, explain briefly.\n"
                "<|channel|>final<|message|>\n"
                "1 2 3\n4 5 6\n7 8 9\n"
            )),
        ),
        Message.from_role_and_content(Role.USER, TextContent(text=user_text)),
    ])

    # Render for completion (assistant start) and feed token IDs directly
    tok_ids = harmony.render_conversation_for_completion(convo, Role.ASSISTANT)
    input_ids = torch.tensor([tok_ids], dtype=torch.long, device=model.device)

    # Build stopping sequences
    stop_id_sequences: List[List[int]] = []
    for t in stop_tokens:
        try:
            ids = tok.encode(t, add_special_tokens=False)
            if isinstance(ids, list) and ids:
                stop_id_sequences.append(ids)
        except Exception:
            pass

    # Greedy decoding for structure stability
    out = model.generate(
        input_ids,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=None,  # rely on stop sequences for structured stopping
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        stopping_criteria=StoppingCriteriaList([StopOnSequences(stop_id_sequences)]) if stop_id_sequences else None,
    )

    gen = tok.decode(out[0], skip_special_tokens=False)
    completion = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=False)

    has_analysis = "<|channel|>analysis" in completion
    has_final = "<|channel|>final" in completion
    parsed = parse_grid_from_response(completion, prob.test_pairs[0].y.shape)

    results = {
        "gpu": 3,
        "problem_uid": prob.uid,
        "has_analysis": has_analysis,
        "has_final": has_final,
        "has_grid": parsed is not None,
        "grid_shape": tuple(parsed.shape) if parsed is not None else None,
        "preview": completion[:600],
    }

    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"singleturn_gpu3_{prob.uid}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out_dir / f"singleturn_gpu3_{prob.uid}.txt").write_text(completion)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
