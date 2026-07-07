# LeRobot — Distributed & Large-Model Training Learning Project

> **This is a fork of Hugging Face [LeRobot](https://github.com/huggingface/lerobot) that I used as a
> hands-on learning project for distributed training, GPU profiling, and efficient foundation-model
> fine-tuning.** Everything under the "My Contributions" section is **my work**; everything else in
> the repo is unmodified upstream LeRobot (its original README is preserved as
> [`README_UPSTREAM.md`](./README_UPSTREAM.md)).

I took the **SmolVLA** robot foundation model (450M params) and built + measured an efficient training
pipeline for it on rented **[Modal](https://modal.com) GPUs** — profiling the bottlenecks, adding
efficiency features, fine-tuning it cheaply with LoRA, and scaling it across multiple GPUs.

---

## 🟢 My Contributions

### 1. A learning project with measured results → [`my_contributions/`](./my_contributions/)

| Milestone | What I did | Headline result |
|-----------|-----------|-----------------|
| **M1 — Profiling** | Added `torch.profiler` to the training loop, profiled SmolVLA on a GPU | **GPU idle ~72% of the time** — bottleneck was CPU-side data prep, not compute |
| **M2 — Efficiency** | Added gradient accumulation; wired up mixed precision | Effective batch 32 on **~45% less memory** |
| **M3 — LoRA fine-tuning** | Fine-tuned the pretrained `smolvla_base` with LoRA | Trained just **0.16% of params** (743K of 450M) and it still learned |
| **M4 — Distributed** | Ran 1-GPU vs DDP vs FSDP on 2× A10G | **DDP ~1.9× throughput** (near-linear); real bf16 **+40%**; debugged FSDP |

Full write-ups (with numbers, caveats, and the reasoning): the [`my_contributions/`](./my_contributions/)
folder — one `M*_FINDINGS.md` per milestone, plus [`LEARNING_ROADMAP.md`](./my_contributions/LEARNING_ROADMAP.md).

The [`my_contributions/modal/`](./my_contributions/modal/) folder has the exact **Modal scripts** I used
to run each milestone on rented GPUs.

### 2. Real changes to the LeRobot training pipeline

I modified **two core files** in the upstream source (they must stay in `src/lerobot/` for the package
to run). A self-contained diff of these edits lives in
[`my_contributions/code_changes/`](./my_contributions/code_changes/). In the live files, every added
block is tagged with `[learning-project]` — run `grep -rn "\[learning-project\]" src/` to see all 10.

| File | What I added |
|------|-------------|
| [`src/lerobot/scripts/lerobot_train.py`](./src/lerobot/scripts/lerobot_train.py) | • `torch.profiler` tracing (`LEROBOT_PROFILE=1`) — M1<br>• Gradient accumulation via `accelerator.accumulate()` — M2<br>• Wired `use_amp` into **training** mixed precision (it was inference-only — a real bug I found in M4)<br>• FSDP compatibility (uniform-dtype cast + DDP-kwarg gating) — M4<br>• `M2_BENCH` peak-memory / step-time logging |
| [`src/lerobot/configs/train.py`](./src/lerobot/configs/train.py) | • New `gradient_accumulation_steps` config field — M2 |

### What I learned (the skills this project targeted)

- **Profiling** with `torch.profiler` — reading GPU-utilization / bottleneck traces.
- **Efficient training** — mixed precision, gradient accumulation.
- **Foundation-model fine-tuning** — LoRA / parameter-efficient fine-tuning (PEFT).
- **Distributed training** — data parallelism (DDP) and sharded training (FSDP/ZeRO), and *when each applies*.
- Debugging real distributed failures (mixed-dtype FSDP crashes, an unwired AMP flag).

---

## Reproducing my runs

```bash
pip install modal && modal setup                 # one-time Modal account link
modal run my_contributions/modal/modal_m1.py     # M1: profile SmolVLA
modal run my_contributions/modal/modal_m2.py     # M2: efficiency benchmark
modal run my_contributions/modal/modal_m3.py     # M3: LoRA fine-tuning
modal run my_contributions/modal/modal_m4.py     # M4: DDP vs FSDP (needs 2 GPUs)
```

---

## About the base project (upstream LeRobot)

Everything outside `my_contributions/` and the two tagged files above is unmodified Hugging Face
LeRobot — a PyTorch library for real-world robotics (datasets, pretrained policies, training/eval
tools). Its original documentation is in [`README_UPSTREAM.md`](./README_UPSTREAM.md), and the source
project lives at https://github.com/huggingface/lerobot

_Note: this fork's history was flattened to a single clean commit and the upstream git-LFS test
fixtures (~3.5 GB of videos/tensors) were removed, since they aren't relevant to the learning project._
