# Repository Guidelines

## Project Structure & Module Organization
- `main_*.py`: Training and experimentation entry points (e.g., `main_hf_trl_dapo.py`).
- `test_*.py`: Test scripts in the project root (some perform heavy GPU/model ops).
- `visualizations/`: Plotting and comparison utilities.
- `run_gpt_oss_dapo.sh`: Convenience script for common runs.
- `requirements.txt`: Python dependencies.
- `wandb/`, `*.log`, `*results*.json`: Run artifacts and logs (do not commit large outputs).

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run training: `python main_hf_trl_dapo.py --help` (or `bash run_gpt_oss_dapo.sh`).
- Quick tests: `pytest -q` (use `-k name` to filter; some tests may download models and require a GPU).
- Lint/format (optional if installed): `ruff check .` and `black .`

## Project Memory (Current Defaults)
- Env: `gptoss` (torch/transformers/numpy/arc + openai_harmony)
- GPU: `CUDA_VISIBLE_DEVICES=5`
- Inference: `python scripts/infer_gpu5_harmony_oss_fixed.py` (max_new_tokens=4092)
- Prompting: Use OpenAI Harmony encoding (no manual token injection). Tests and scripts must not force `<|channel|>final<|message|>`.
- Canonical tests: `test_harmony_library.py` (no forced final), `test_harmony_proper.py`, `test_inference.py` (lightweight).
- Logs: `logs/inference_harmony_oss_fixed_*.{json,txt}` include has_analysis/has_final, grid_extracted, accuracy.

## Notes on Cleanup
- Deprecated patterns: any test or script that manually appends `<|channel|>final<|message|>` to inputs. Updated tests avoid forced insertion.
- Prefer Harmony-rendered inputs via `openai_harmony` and Harmony stop sequences. Fall back to chat templates only if Harmony is unavailable.

## Coding Style & Naming Conventions
- Python (3.10+), PEP 8, 4-space indentation.
- Modules/files and functions: `snake_case.py` / `snake_case`.
- Classes: `CapWords`; constants: `UPPER_SNAKE_CASE`.
- Tests live as `test_*.py` near related scripts; prefer small, deterministic helpers.
- Do not commit large artifacts (models, `.pt`, `wandb/` runs, large logs).

## Testing Guidelines
- Framework: `pytest`. Name tests `test_<unit>.py::test_<behavior>`.
- Prefer unit tests around data prep, prompting, and utilities; isolate GPU/internet-heavy tests or guard with env flags.
- Run locally: `pytest -q -k <subset>`; save any diagnostic outputs to temporary files (not versioned).

## Commit & Pull Request Guidelines
- Messages: Conventional Commits style, e.g. `feat: add GRPO LoRA flag`, `fix: guard CUDA device map`, `test: add inference settings sweep`.
- PRs must include: clear summary, motivation, reproduction commands, before/after notes (logs or screenshots), and linked issues.
- Keep diffs focused; update `README.md` when changing flags, scripts, or workflows.

## Security & Configuration Tips
- Configure environment variables: `CUDA_VISIBLE_DEVICES` to pick GPUs; `WANDB_API_KEY` for tracking. Never commit secrets.
- Be mindful that some tests/scripts pull remote models; pin versions in flags where possible and document them in PRs.
