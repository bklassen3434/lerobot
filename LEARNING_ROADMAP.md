# Learning Roadmap — Efficient Training Pipeline for a Robot Foundation Model

**Goal:** learn distributed training, profiling, foundation-model fine-tuning, and efficient
training — and produce one impressive, demoable artifact for job interviews.

**The capstone:** take **SmolVLA** (a real vision-language-action foundation model in this repo),
and build a *measured* pipeline to fine-tune it efficiently — profile it, optimize it, fine-tune it
cheaply with LoRA, and scale it across GPUs.

**The one-sentence interview story:**
> "I took an open-source robot foundation model and built an efficient fine-tuning pipeline for it —
> profiled the bottlenecks, cut training memory with gradient accumulation and FSDP, fine-tuned it
> with LoRA on a single rented GPU for under $10, and have the benchmarks and a working demo to show it."

**The deliverable (the "cool output"):** a GitHub fork containing (1) a benchmark report README with
before/after charts, (2) a demo GIF of the fine-tuned policy, (3) clean commits showing the work.

---

## Milestones (do in order; strong version = M1–M4)

- [x] **M1 — Baseline + profile it.** Train SmolVLA on `lerobot/pusht`, add `torch.profiler`, find the bottleneck.
      - Artifact: profiler trace + findings → `M1_FINDINGS.md` (trace: `outputs_from_modal/smolvla_m1_trace.json`).
      - **Result: GPU idle 72% of the time (28% util); bottleneck is CPU-side data prep + FP32 attention.**
      - Skill: profiling. ✅ Ran on Modal A10G via `modal_m1.py`.
- [x] **M2 — Make it efficient.** Mixed precision (`--policy.use_amp=true`) + gradient accumulation (added to the code), benchmarked.
      - Artifact: before/after table → `M2_FINDINGS.md` (ran via `modal_m2.py`).
      - **Result: AMP ~13% faster; grad accumulation reaches effective batch 32 on 45% less memory (2735 vs 4958 MB).**
      - Skill: mixed precision + gradient accumulation. ✅ Added `gradient_accumulation_steps` config + `accelerator.accumulate()` wiring.
- [x] **M3 — Fine-tune cheaply with LoRA.** Loaded pretrained `smolvla_base`, fine-tuned full vs LoRA.
      - Artifact: full-vs-LoRA comparison → `M3_FINDINGS.md` (ran via `modal_m3.py`). Demo GIF still TODO (needs a longer real training run + eval env).
      - **Result: LoRA trained just 0.16% of params (743K vs 100M, 134× fewer) and the model still learned; ~19% less GPU memory.**
      - Skill: parameter-efficient foundation-model fine-tuning (LoRA/PEFT). ✅
- [x] **M4 — Scale across GPUs (DDP → FSDP).** 1-GPU (fp32/bf16), 2-GPU DDP, 2-GPU FSDP on 2× A10G.
      - Artifact: results + analysis → `M4_FINDINGS.md` (ran via `modal_m4.py`).
      - **Result: DDP ~1.89× throughput on 2 GPUs (near-linear); real bf16 +40% throughput; FSDP got running but costs more than it saves at this model size — the lesson being FSDP is for models that don't fit one GPU.**
      - Also fixed a real bug: `use_amp` was never wired into training (only inference) — now it is.
      - Skill: distributed training (DDP + FSDP), the most marketable item on the list. ✅
- [ ] **M5 (optional) — World models writeup.** Explain how this repo's TDMPC (model-based RL) differs from imitation policies.
      - Artifact: "how world models work, in code" README section.
      - Skill: RL / world-models research-domain credibility.

## Prioritization

| Time | Do | You can say… |
|------|----|--------------|
| 2 weeks | M1 + M2 | "I profiled and optimized a real training pipeline, with benchmarks" |
| 1 month | M1–M3 | + "I fine-tuned a foundation model cheaply with LoRA — here's the demo" |
| 6–8 weeks | M1–M4 | + "I scaled it across GPUs with FSDP" ← the strong version |
| +1 week | add M5 | + research-domain credibility (world models / RL) |

## Compute

- All milestones run on a rented GPU. **Modal** (serverless, per-second billing) is the safe default —
  no risk of leaving an idle box running. Cheap SSH boxes (RunPod / vast.ai) are fine for interactive M1 profiling.
- Set a spending limit / budget alert before starting.

## Key files in this repo

| Thing | File | Where |
|-------|------|-------|
| Training loop | `src/lerobot/scripts/lerobot_train.py` | `update_policy()` ~L65–160; main loop ~L457–516 |
| AMP flag | `src/lerobot/configs/policies.py` | `use_amp: bool = False` |
| LoRA config | `src/lerobot/configs/default.py` | `PeftConfig` ~L95–126 |
| LoRA wiring | `src/lerobot/policies/pretrained.py` | `wrap_with_peft()` ~L270–317 |
| SmolVLA model | `src/lerobot/policies/smolvla/modeling_smolvla.py` | `_get_default_peft_targets()` |
| World model (RL) | `src/lerobot/policies/tdmpc/modeling_tdmpc.py` | dynamics + planning |
