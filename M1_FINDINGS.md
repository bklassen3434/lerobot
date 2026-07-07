# M1 — Baseline + Profile (SmolVLA)

**Setup:** fresh SmolVLA (450M total / 100M trainable params), `lerobot/pusht` (3 episodes),
batch size 8, on a rented **NVIDIA A10G** via Modal. Profiled 5 steps with `torch.profiler`
(enabled by `LEROBOT_PROFILE=1`). Trace: `outputs_from_modal/smolvla_m1_trace.json`
(open in `chrome://tracing`).

## Headline finding

> **The GPU sits idle ~72% of the time.** Each training step takes ~558 ms of wall-clock,
> but the GPU is only doing math for ~158 ms of it (28% utilization). The step is bottlenecked
> on **CPU-side work — data prep and dtype/device conversions — not GPU compute.**

| Metric | Value |
|--------|-------|
| Avg wall time / step | 557.7 ms |
| GPU busy / step | 158.4 ms |
| **GPU utilization** | **28%** |
| **GPU idle** | **72%** |
| GPU memcpy/memset | 8.2 ms (tiny — transfers aren't the problem) |

## Where the GPU time goes (when it *is* working)

| GPU kernel | Time | What it is |
|-----------|------|------------|
| `fmha_cutlassF_f32...AttentionKernel` | 197 ms | **Attention, running in FP32** ← biggest single cost |
| `cutlass...s1688gemm` (×2) | 228 ms | Matmuls (the transformer/VLM linear layers) |
| `GeluCUDAKernel`, layernorm, elementwise | ~90 ms | Activations / norms |

Note the top kernel is **FP32** attention — the model is doing its most expensive GPU op in
full precision. That's a direct target for M2 (mixed precision).

## Where the CPU time goes (the real bottleneck)

The most expensive CPU-side ops were `aten::to` / `aten::_to_copy` / `aten::copy_`
(~600 ms cumulative) — i.e. **tensor dtype/device conversions**. Combined with the 72% idle
GPU, this says the step is dominated by getting data *ready* for the GPU, not by the GPU itself.
(One concrete culprit lives in the training loop: images are converted `uint8 → float32` on the
CPU every step, at `lerobot_train.py:460-462`.)

## What this sets up for M2

Two clear levers, both measurable against the numbers above:
1. **Mixed precision (`--policy.use_amp=true`)** → should shrink the 197 ms FP32 attention kernel
   and cut conversion overhead. Watch peak memory drop too.
2. **Feed the GPU faster** → more dataloader workers / move the uint8→float conversion, to close
   the 72% idle gap. The win here is potentially larger than #1, since the GPU is starving.

**Baseline numbers to beat in M2:** 557.7 ms/step wall, 28% GPU utilization.

_Caveat: tiny dataset (3 episodes), only 5 profiled steps, fresh weights — good enough as a
learning baseline. On a larger dataset the data-loading bottleneck typically gets worse, not better._
