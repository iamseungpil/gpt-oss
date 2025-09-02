#!/usr/bin/env python3
"""
GPU5 inference smoke test (oss_fixed env) using OpenAI Harmony encoding.

Verifies:
- Harmony channels: analysis → final
- ARC-format grid extraction from final channel
- Uses same conversation rendering approach as main_hf_trl_dapo.py

Run (on host with proper env):
  conda activate oss_fixed
  CUDA_VISIBLE_DEVICES=5 python scripts/infer_gpu5_harmony_oss_fixed.py
"""

import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from arc import train_problems

try:
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
    HAVE_HARMONY = True
except Exception:
    HAVE_HARMONY = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("infer_gpu5_harmony")


def grid_to_string(grid: np.ndarray) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def parse_grid_block(text: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    lines = text.strip().split("\n")
    rows: List[List[int]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(h in line.lower() for h in ["grid:", "output:", "answer:", "result:", "input:", "example", "channel", "message"]):
            continue
        parts = line.replace(",", " ").split()
        vals: List[int] = []
        for p in parts:
            try:
                n = int(float(p)) if "." in p else int(p)
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
            # Stop if we already started collecting grid and hit an invalid line
            break
    if rows:
        width_ok = len({len(r) for r in rows}) == 1 and len(rows[0]) == target_shape[1]
        height_ok = len(rows) == target_shape[0]
        if width_ok and height_ok:
            return np.array(rows)
    return None


def parse_grid_from_response(response: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    # Prefer final channel
    m = re.search(r"<\|channel\|>final<\|message\|>([\s\S]*?)(?:<\|end\|>|$)", response)
    if m:
        blk = m.group(1).strip()
        grid = parse_grid_block(blk, target_shape)
        if grid is not None:
            return grid
    # Fallback: code blocks
    for blk in re.findall(r"```(?:python|text)?\n?([\s\S]*?)```", response):
        grid = parse_grid_block(blk.strip(), target_shape)
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


def build_harmony_prompt_tokens_or_text(problem, tokenizer: AutoTokenizer):
    examples = []
    for i, tr in enumerate(problem.train_pairs, 1):
        examples.append(f"Example {i}:\nInput:\n{grid_to_string(tr.x)}\nOutput:\n{grid_to_string(tr.y)}")
    examples_text = "\n\n".join(examples)
    test_input = problem.test_pairs[0].x

    user_content = (
        "Solve this ARC puzzle:\n\n"
        + examples_text
        + "\n\nTest Input:\n"
        + grid_to_string(test_input)
        + "\n\nAnalyze the pattern, then provide the solution grid."
    )

    if HAVE_HARMONY:
        harmony = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(
                    Role.SYSTEM,
                SystemContent(
                    text="You are ChatGPT, a large language model trained by OpenAI.",
                    knowledge_cutoff="2024-06",
                    reasoning="high",
                ),
                ),
                Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent(
                        text=(
                            "# ARC Puzzle Solver\n\n"
                            "You solve ARC puzzles by:\n"
                            "1. Analyzing patterns in training examples\n"
                            "2. Identifying transformation rules\n"
                            "3. Applying the rule to the test input\n\n"
                            "Use the analysis channel for reasoning, then the final channel for your grid solution.\n\n"
                            "Example format:\n"
                            "<|channel|>analysis<|message|>\n"
                            "Pattern identified: Fill empty cells with 1s, keep original values.\n"
                            "<|channel|>final<|message|>\n"
                            "1 2 3\n4 5 6\n7 8 9\n"
                        ),
                    ),
                ),
                Message.from_role_and_content(Role.USER, TextContent(text=user_content)),
            ]
        )
        token_ids = harmony.render_conversation_for_completion(convo, Role.ASSISTANT)
        return {"token_ids": token_ids, "harmony": harmony}
    else:
        # Fallback: use chat template without harmony lib
        messages = [
            {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI.", "reasoning": {"effort": "high"}},
            {"role": "developer", "content": (
                "# ARC Puzzle Solver\n"
                "Use the analysis channel for reasoning, then the final channel for your grid solution.\n"
                "Grid: digits 0-9, space-separated, one row per line. No extra text in final.\n\n"
                "Example format:\n"
                "<|channel|>analysis<|message|>\n"
                "Identify the rule briefly.\n"
                "<|channel|>final<|message|>\n"
                "1 2 3\n4 5 6\n7 8 9\n"
            )},
            {"role": "user", "content": user_content},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"input_text": input_text}


def main():
    # Respect external CUDA_VISIBLE_DEVICES (e.g., run with GPU 5)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

    model_name = "openai/gpt-oss-20b"
    logger.info("Loading tokenizer and model...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},  # maps to first visible GPU (GPU5)
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        # Fallback without FA2 if not available in this env
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    model.eval()
    logger.info("Model ready.")

    # Problem selection
    prob = train_problems[0]
    target_grid = prob.test_pairs[0].y
    target_shape = target_grid.shape

    # Build Harmony-rendered tokens (preferred) or chat template text (fallback)
    built = build_harmony_prompt_tokens_or_text(prob, tok)
    stopping_criteria = None
    input_ids = None
    eos_ids = tok.eos_token_id
    if "token_ids" in built:
        tok_ids = built["token_ids"]
        input_ids = torch.tensor([tok_ids], dtype=torch.long, device=model.device)
        # Stop sequences from Harmony encoding
        harmony = built.get("harmony")
        stop_tokens = harmony.stop_tokens_for_assistant_actions() if harmony else []
        stop_id_sequences: List[List[int]] = []
        for t in stop_tokens:
            try:
                ids = tok.encode(t, add_special_tokens=False)
                if isinstance(ids, list) and ids:
                    stop_id_sequences.append(ids)
            except Exception:
                pass
        stopping_criteria = StoppingCriteriaList([StopOnSequences(stop_id_sequences)]) if stop_id_sequences else None
    else:
        # Fallback: tokenize chat template text
        input_text = built["input_text"]
        enc = tok(input_text, return_tensors="pt")
        input_ids = enc.input_ids.to(model.device)
        # Add <|return|> as extra EOS if available (200002)
        eos_ids = [tok.eos_token_id, 200002] if tok.eos_token_id is not None else 200002

    logger.info("Generating with max_new_tokens=10000...")
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=10000,
            temperature=0.5,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos_ids,
            stopping_criteria=stopping_criteria,
        )

    completion = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=False)
    has_analysis = "<|channel|>analysis" in completion
    has_final = "<|channel|>final" in completion
    parsed = parse_grid_from_response(completion, target_shape)

    results = {
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        "problem_uid": prob.uid,
        "max_new_tokens": 4092,
        "has_analysis_channel": has_analysis,
        "has_final_channel": has_final,
        "grid_extracted": parsed is not None,
        "grid_shape": parsed.shape if parsed is not None else None,
        "target_shape": list(target_shape),
        "accuracy": float(np.mean(parsed == target_grid)) if parsed is not None and parsed.shape == target_shape else None,
        "preview": completion[:800],
        "harmony_library_used": HAVE_HARMONY,
    }

    out_dir = Path("logs"); out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"inference_harmony_oss_fixed_{ts}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out_dir / f"inference_harmony_oss_fixed_{ts}.txt").write_text(completion)

    logger.info(json.dumps({k: v for k, v in results.items() if k not in ("preview",)}, indent=2, ensure_ascii=False))
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
