# Learning Project — Efficient Training of a Robot Foundation Model

My hands-on project on top of LeRobot (see the [repo root README](../README.md) for the overview).
Everything here is my own work.

## Contents

- **[LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md)** — the plan and milestone checklist (M1–M4, all done).
- **[M1_FINDINGS.md](./M1_FINDINGS.md)** — profiling: the GPU sat idle ~72% of the time.
- **[M2_FINDINGS.md](./M2_FINDINGS.md)** — efficiency: gradient accumulation → effective batch 32 on ~45% less memory.
- **[M3_FINDINGS.md](./M3_FINDINGS.md)** — LoRA: fine-tuned `smolvla_base` training just 0.16% of params.
- **[M4_FINDINGS.md](./M4_FINDINGS.md)** — distributed: DDP ~1.9× scaling, real bf16 +40%, FSDP debugged.
- **[modal/](./modal/)** — the Modal scripts that ran each milestone on rented GPUs.

## The pipeline code I changed

Lives in the LeRobot source (outside this folder), tagged with `[learning-project]`:
- `src/lerobot/scripts/lerobot_train.py` — profiler, gradient accumulation, AMP wiring, FSDP support.
- `src/lerobot/configs/train.py` — the `gradient_accumulation_steps` config field.

Find every change with: `grep -rn "\[learning-project\]" ../src/`
