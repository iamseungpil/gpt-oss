# Repository Guidelines

## Project Structure & Module Organization
- `main_*.py`: Training/experiments entry points (e.g., `main_hf_trl_dapo.py`).
- `test_*.py`: Tests at repo root; some are GPU/model-heavy.
- `visualizations/`: Plotting and comparison utilities.
- `scripts/`: Inference and helpers (e.g., `scripts/infer_gpu5_harmony_oss_fixed.py`).
- `run_gpt_oss_dapo.sh`: Convenience script for common runs.
- `requirements.txt`: Python dependencies.
- Do not commit large artifacts: `wandb/`, `*.pt`, big logs (`*results*.json`).

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Train/inspect flags: `python main_hf_trl_dapo.py --help` or `bash run_gpt_oss_dapo.sh`
- Quick tests: `pytest -q` (filter: `pytest -q -k name`)
- Lint/format (optional): `ruff check .` and `black .`
- Inference (defaults): `CUDA_VISIBLE_DEVICES=5 python scripts/infer_gpu5_harmony_oss_fixed.py`

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8, 4-space indentation.
- Files/functions: `snake_case.py` / `snake_case`; classes: `CapWords`; constants: `UPPER_SNAKE_CASE`.
- Keep changes minimal and focused; update `README.md` when changing flags/scripts.

## Testing Guidelines
- Framework: `pytest`. Name tests `test_<unit>.py::test_<behavior>`.
- Canonical lightweight checks: `test_harmony_library.py`, `test_harmony_proper.py`, `test_inference.py`.
- Some tests may download models or require a GPU. Use `-k` to subset locally; isolate heavy tests behind env flags when possible.

## Commit & Pull Request Guidelines
- Conventional Commits, e.g., `feat: add GRPO LoRA flag`, `fix: guard CUDA device map`, `test: add inference settings sweep`.
- PRs include: clear summary, motivation, reproduction commands, before/after notes (logs or screenshots), and linked issues.
- Keep diffs focused; document new flags, scripts, or workflows.

## Security & Configuration Tips
- Set env vars: `CUDA_VISIBLE_DEVICES` to select GPUs; `WANDB_API_KEY` for tracking. Never commit secrets.
- Prompting: use OpenAI Harmony encoding via `openai_harmony`; prefer Harmony stop sequences. Do NOT manually inject `<|channel|>final<|message|>`.
- Remote models may be pulled; pin versions in flags and note them in PRs.
- Logs: `logs/inference_harmony_oss_fixed_*.{json,txt}` capture metrics; keep sizes small and exclude large artifacts from commits.

