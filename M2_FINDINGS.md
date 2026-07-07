# M2 — Efficiency: Mixed Precision + Gradient Accumulation (SmolVLA)

> **⚠️ CORRECTION (found in M4):** the "mixed precision ~13% faster" result below was **measurement
> noise**, not a real effect. It turned out `--policy.use_amp` was only wired into *inference*, never
> into the training loop — so both "fp32" and "amp" runs here actually ran in fp32. In M4 we fixed the
> wiring (training now really runs bf16 when `use_amp=true`) and measured the **true AMP effect: ~40%
> higher throughput** (see `M4_FINDINGS.md`, A0 vs A1). The gradient-accumulation result below is
> unaffected and still valid.



**Setup:** fresh SmolVLA (450M params), `lerobot/pusht` (3 episodes), on a rented **NVIDIA A10G**
(24 GB) via Modal (`modal_m2.py`). Each config trains 12 steps; we log peak GPU memory and average
step time via the `M2_BENCH` line added to `lerobot_train.py`.

## Raw results

| Config | batch | grad_accum | AMP | effective batch | avg step | peak GPU mem |
|--------|-------|-----------|-----|-----------------|----------|--------------|
| 1. baseline (fp32) | 8 | 1 | ✗ | 8 | 524 ms | 2512 MB |
| 2. + mixed precision | 8 | 1 | ✓ | 8 | **458 ms** | 2512 MB |
| 3. big batch (naive) | 32 | 1 | ✓ | 32 | 835 ms | **4958 MB** |
| 4. + grad accumulation | 8 | 4 | ✓ | 32 | 431 ms/micro | **2735 MB** |

## What each optimization bought

**Mixed precision (config 1 → 2, same effective batch of 8):**
- **12.6% faster per step** (524 → 458 ms).
- Peak memory unchanged here (2512 MB). At this small scale the allocator peak is dominated by
  fixed costs, so AMP's memory benefit is negligible — its win is *speed*. (On larger models/batches
  AMP also cuts memory noticeably.)

**Gradient accumulation — the headline (config 3 vs 4, both effective batch 32):**
- Reaching an effective batch of 32 the **naive** way (a real batch of 32) costs **4958 MB**.
- Reaching the **same** effective batch of 32 via batch-8 × accum-4 costs only **2735 MB**.
- **→ 45% less GPU memory for the same effective batch size.**
- The tradeoff (important, and worth saying out loud in an interview): grad accumulation does 4
  micro-batches per optimizer update (4 × 431 ≈ 1724 ms) vs. the naive 835 ms single step — so it
  trades **~2× compute time for ~45% less memory**. That's exactly the trade you make when you're
  memory-bound and can't fit the batch you want.

## The one-line takeaways

- Mixed precision: **~13% faster**, free (one flag).
- Gradient accumulation: lets you train at a **large effective batch on ~half the memory** a real
  batch would need — the standard trick for fitting big-batch training onto a smaller GPU.

_Caveat: SmolVLA at 2.5 GB is tiny next to the A10G's 24 GB, so we're nowhere near OOM — that's why
AMP's memory savings look flat. The value shown here is the memory **scaling** (2735 vs 4958 MB for
identical effective batch), which is what matters when the model actually fills the GPU._

## Verification that grad accumulation is correct

Local CPU smoke test (ACT, batch 2, accum 3) showed the optimizer stepping exactly every 3rd
micro-batch: grad-norm was computed on steps 3 and 6 only, and zero on the accumulating steps in
between — confirming gradients accumulate across micro-batches and the optimizer updates once per
group.
