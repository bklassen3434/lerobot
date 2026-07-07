# My code changes to the LeRobot training pipeline

My contributions include edits to **two files in the upstream LeRobot source**. Those files have to
stay in `src/lerobot/` for the Python package to work, so they can't be moved here — but this folder
gives you a **self-contained record** of exactly what I changed.

- **[`lerobot_training_pipeline.patch`](./lerobot_training_pipeline.patch)** — a git diff of all my
  edits versus the upstream base commit (`9db9c35c`). View it here, or apply it to a clean LeRobot
  checkout with `git apply lerobot_training_pipeline.patch`.

## The two files I changed (live versions in `src/`)

Every added block is tagged `[learning-project]`. Find them with:
```bash
grep -rn "\[learning-project\]" ../../src/
```

| File | What I added | Milestone |
|------|-------------|-----------|
| `src/lerobot/scripts/lerobot_train.py` | `torch.profiler` tracing (`LEROBOT_PROFILE=1`) | M1 |
| | Gradient accumulation via `accelerator.accumulate()` | M2 |
| | Wired `use_amp` into **training** mixed precision (was inference-only — a real bug I found) | M2/M4 |
| | FSDP compatibility (uniform-dtype cast + DDP-kwarg gating) | M4 |
| | `M2_BENCH` peak-memory / avg-step-time logging | M2/M4 |
| `src/lerobot/configs/train.py` | New `gradient_accumulation_steps` config field | M2 |

Diff summary: **+147 / −42 lines across the two files.**
