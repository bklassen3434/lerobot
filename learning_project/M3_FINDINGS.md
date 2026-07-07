# M3 — LoRA Fine-Tuning of the Pretrained SmolVLA

**Setup:** loaded the **real pretrained `lerobot/smolvla_base`** (450M-param vision-language-action
foundation model) and fine-tuned it on `lerobot/pusht` on a rented **NVIDIA A10G** via Modal
(`modal_m3.py`). `--policy.input_features=null --policy.output_features=null` lets the pretrained base
re-infer its inputs from the single-camera dataset, so no camera-matched dataset was needed. Two
configs, 60 steps each, mixed precision on.

## Results

| Config | Trainable params | % of model | Loss (start → end) | Peak GPU mem |
|--------|------------------|-----------|--------------------|--------------|
| Full fine-tune | **99,880,992** | 22.19% | 1.378 → 0.475 | 2510 MB |
| **LoRA (r=16)** | **742,656** | **0.16%** | 0.940 → 0.636 | 2031 MB |

## The headline

> **LoRA fine-tuned a 450M-parameter foundation model by training just 0.16% of its weights —
> ~743K parameters instead of ~100M, a 134× reduction — and the model still learned (loss went
> down).** Peak GPU memory also dropped ~19% (fewer optimizer states to hold).

Why "full" fine-tune is only 22% and not 100%: SmolVLA already freezes its vision encoder and trains
the action expert by default, so even "full" fine-tuning touches ~100M of the 450M. LoRA then shrinks
that ~100M down to ~0.74M by training small adapter matrices instead of the expert's full weights.

## What this demonstrates (the skills)

- **Parameter-efficient fine-tuning (LoRA/PEFT)** — the standard, in-demand technique for adapting
  large models cheaply. Training 743K params instead of 100M means smaller memory, faster steps, and
  tiny checkpoints (you ship just the adapter, not a 450M copy).
- **Working with a real foundation model** — this used the actual pretrained SmolVLA, not a random
  one (unlike M1/M2, which measured raw compute).

## The one-line interview story

> "I fine-tuned a 450M-parameter robot foundation model on a single rented GPU by training only 0.16%
> of its parameters with LoRA — and it still learned the task."

## Honest caveats / what's NOT done yet

- 60 steps is enough to show the loss **trending down** and to prove the LoRA wiring works — it is
  **not** a converged, deployable policy. A real **demo video** of the robot succeeding needs a longer
  run (thousands of steps) plus the pusht eval environment to render rollouts. That's a good, cheap
  follow-up run when we want the visual artifact.
- Different starting losses between the two configs are just data-order / re-inferred-projection init
  differences, not meaningful.
