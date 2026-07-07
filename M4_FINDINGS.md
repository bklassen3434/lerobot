# M4 — Distributed Training: 1-GPU vs DDP vs FSDP (SmolVLA)

**Setup:** fresh SmolVLA (450M params), `lerobot/pusht`, on a **2× NVIDIA A10G** Modal container
(`modal_m4.py`). Each config trains 30 steps, batch 8 per GPU. We log peak per-GPU memory and step
time (`M2_BENCH`) and compute throughput = effective_batch / step_time.

## Results

| Config | Launcher | per-GPU mem | avg step | effective batch | throughput |
|--------|----------|-------------|----------|-----------------|------------|
| A0. 1 GPU, fp32 | `lerobot-train` | 2512 MB | 308 ms | 8 | 26.0 samples/s |
| A1. 1 GPU, **bf16** | `lerobot-train` | 2436 MB | 221 ms | 8 | **36.3 samples/s** |
| B. 2 GPU, **DDP**, fp32 | `torchrun --nproc_per_node=2` | 2713 MB | 326 ms | 16 | **49.1 samples/s** |
| C. 2 GPU, **FSDP**, fp32 | `accelerate launch --use_fsdp` | 9065 MB | 786 ms | 16 | 20.4 samples/s |

## What each comparison shows

**Mixed precision (A0 → A1) — +40% throughput.**
Real bf16 (now correctly wired into training — see the note below) lifted throughput from 26.0 to
36.3 samples/s (+40%) with slightly less memory. This is the *true* AMP effect; M2's earlier "13%"
was measurement noise because the flag wasn't actually connected to training.

**Data parallelism / DDP (A0 → B) — near-linear scaling.**
Two GPUs raised throughput from 26.0 to 49.1 samples/s = **1.89× on 2 GPUs** (near-ideal). Each GPU
holds a full copy of the model, so per-GPU memory is essentially unchanged (2512 → 2713 MB). This is
the workhorse of multi-GPU training and it "just works" here.

**Sharded training / FSDP (B → C) — the nuanced lesson.**
FSDP used **more** memory (9065 MB) and ran **slower** (20.4 samples/s) than DDP. That is the correct,
expected outcome *at this scale*, for two reasons:
1. **SmolVLA isn't natively FSDP-friendly.** It keeps its VLM backbone in bf16 while the rest is fp32;
   FSDP requires a uniform dtype per flattened unit, so we had to cast the whole model to fp32
   (removing SmolVLA's own memory optimization). It also accesses some weights outside their owning
   submodule, which breaks per-layer wrapping — so we had to use `NO_WRAP` (whole model gathered each
   step), which limits FSDP's savings.
2. **The model already fits on one GPU.** FSDP shards the model to fit training that *doesn't* fit on a
   single GPU. Forcing it on a 2.5 GB model that fits comfortably in 24 GB is pure overhead — you pay
   the communication and full-gather cost with no sharding benefit to offset it.

> **The real takeaway (and the better interview answer):** DDP is the right tool when the model fits;
> FSDP earns its keep only when the model is too big for one GPU. Here I demonstrated DDP scaling
> cleanly, got FSDP running despite SmolVLA's mixed-dtype design, and measured that FSDP is
> counterproductive at this scale — which is exactly *why* you'd reserve it for large models.

## Code changes made in M4

- **Wired `use_amp` into training** (`lerobot_train.py`): the training Accelerator now runs bf16 when
  `--policy.use_amp=true`. Previously the flag only affected inference, so training was always fp32.
- **FSDP compatibility** (`lerobot_train.py`): omit DDP-only kwargs under FSDP, and cast the policy to
  a uniform fp32 dtype before FSDP wrapping (SmolVLA's mixed bf16/fp32 params otherwise crash FSDP).

## Honest caveats

- FSDP's memory number is inflated by the required fp32 cast + NO_WRAP full-gather; it is **not**
  evidence that "FSDP is bad," only that it's the wrong tool for a model this small. A proper FSDP win
  would need a model that exceeds single-GPU memory and a transformer-aware wrap policy.
- 30 steps, tiny dataset — these measure throughput/memory, not model quality.
