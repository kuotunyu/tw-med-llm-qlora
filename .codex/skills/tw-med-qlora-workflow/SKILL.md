---
name: tw-med-qlora-workflow
description: Govern phased implementation, testing, documentation, cost gates, and publication for the tw-med-llm-qlora repository. Use whenever Codex plans, edits, tests, trains, evaluates, documents, resumes, or publishes work in this project, especially across the MedQA data, Colab QLoRA, TMMLU+ evaluation, Windows inference, and Hugging Face adapter phases.
---

# TW Med QLoRA Workflow

Use this workflow to keep the research reproducible across interrupted sessions.

## Start every task

1. Read `AGENTS.md` and `PROJECT_PLAN.md` completely.
2. Identify the active Phase and its exit criteria.
3. Read [references/phase-checklist.md](references/phase-checklist.md).
4. Inspect the current worktree and preserve unrelated user changes.
5. Keep all work inside the active Phase unless the user has approved the next gate.

## Implement safely

- Use Python 3.11 and uv; update `uv.lock` whenever dependencies change.
- Consult Context7 before using an external package API. If unavailable, consult current official documentation and record the fallback.
- Keep model IDs configurable. Keep cloud model strings in `.env` and `.env.example`, never hardcode their secrets.
- Keep `.env`, weights, raw/processed questions, full private generations, checkpoints, and large artifacts out of Git.
- Never use test examples for training, prompt tuning, checkpoint selection, or hyperparameter selection.
- On Windows, use pathlib, supported wheels, PyTorch CUDA, chromadb when vector storage is needed, and Ollama for local LLM services. Do not substitute vLLM on Windows.
- In Colab, detect the GPU before loading a model and use only the approved profile from `configs/project.toml`.
- Estimate paid batch work from a smoke/calibration run and wait for explicit approval before unlocking it.

## Finish every change

1. Run the smallest relevant tests, then the Phase validation suite.
2. Update `PROJECT_PLAN.md` with status, commands, evidence, artifacts, risks, and next action.
3. Update README only with measured results; label unfinished work as not run.
4. Use a focused Conventional Commit. Do not combine unrelated features.
5. At a Phase boundary, show the evidence and wait for user confirmation.
