# Phase checklist

## Shared checks

- Confirm the current Phase in `PROJECT_PLAN.md`.
- Confirm `.env`, data, model weights, checkpoints, and private generations are ignored.
- Record exact dataset/model revisions, seed, hardware, package lock, commands, and outputs.
- Run pytest and any Phase-specific validation.
- Run `uv run pytest -q tests/test_project_skill.py` to validate the repo-local workflow skill.
- Update the execution log before committing.
- Stop at the Phase gate and request confirmation.

## Phase 0

- Validate Python 3.11 and `uv.lock`.
- Run `pytest` and `ruff check .`.
- Verify `git check-ignore .env` succeeds and `.env` is not staged.

## Phase 1

- Run the five-row schema and encoding gate before full download.
- Preserve official splits and enforce `test > validation > train` deduplication.
- Assert no normalized question overlaps remain.
- Commit aggregate counts and hashes, never question text.

## Phase 2

- Detect GPU, BF16 support, and VRAM before model loading.
- Keep full training disabled.
- Check finite loss, peak VRAM, adapter save/reload, and strict A-D parsing.
- Print projected full-run time, compute units, and monetary estimate.

## Phase 3

- Require the recorded user approval before enabling full training.
- Keep test data out of the trainer.
- Verify checkpoint archive integrity and resume behavior.
- Save adapter, tokenizer, trainer state, curves, and run manifest.

## Phase 4

- Calibrate 20 questions before the full evaluation.
- Evaluate original instruct, localized base, and adapter with the same prompts.
- Report parse rate, paired uncertainty, subject tables, and the forgetting margin.
- Keep full generations private; publish only IDs and non-verbatim summaries.

## Phase 5

- Validate Windows CUDA base + adapter inference on the RTX 4090.
- Recheck current GGUF export API before optional conversion.
- Verify model-card licensing and medical-use disclaimers.
- Publish only after the user confirms repository targets and visibility.
