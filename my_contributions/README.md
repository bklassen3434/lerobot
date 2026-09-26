# Learning Project — Efficient Training of a Robot Foundation Model

My hands-on project on top of LeRobot (see the [repo root README](../README.md) for the overview).
Everything here is my own work.

## Contents

- **[PROJECT_LOG.md](./PROJECT_LOG.md)** — **start here.** Every dataset and every trained
  model, what each one refers to, and the strategy behind the sequence.
- **[LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md)** — the plan and milestone checklist (M1–M4, all done).
- **[M1_FINDINGS.md](./M1_FINDINGS.md)** — profiling: the GPU sat idle ~72% of the time.
- **[M2_FINDINGS.md](./M2_FINDINGS.md)** — efficiency: gradient accumulation → effective batch 32 on ~45% less memory.
- **[M3_FINDINGS.md](./M3_FINDINGS.md)** — LoRA: fine-tuned `smolvla_base` training just 0.16% of params.
- **[M4_FINDINGS.md](./M4_FINDINGS.md)** — distributed: DDP ~1.9× scaling, real bf16 +40%, FSDP debugged.
- **[modal/](./modal/)** — the Modal scripts that ran each milestone on rented GPUs.
- **[tools/](./tools/)** — dataset checks, training patches, and evaluation probes for the
  pick-the-named-pen task.

## Progress model

`tools/progress_labels.py` + `tools/progress_model.py` answer "how far along is this frame,
toward picking up *that* pen?" with a number in [0, 1]. It exists because the project had no
automated way to tell whether a rollout worked — only an offline action-delta probe, or a
human watching the arm.

Demonstration labels are derived from the action stream (arm starts moving → gripper closes →
pen is up), never from a clock, never by hand. The model is a frozen SigLIP tower plus a small
MLP head conditioned on the colour word, trained with the wrong colour as a zero-progress
negative.

Demonstrations alone are not enough: every one of them succeeds, so the model scored "the
gripper closed next to the pink pen" at 0.90 even when the pen never left the table. Real
rollouts supply the missing half. Their outcomes — which pen the gripper closed on, and
whether it came up — are checked by eye once and recorded in `rollout_outcomes.json`
(the joints cannot decide it: one episode lifted a pen with 1.1° of shoulder_lift, because the
lift came from the wrist).

    # labels + diagnostic figures
    uv run --no-sync --with matplotlib python my_contributions/tools/progress_labels.py \
        bklassen3434/pick_pen_v2_20260920_124400 --plot outputs/progress_model/labels.png

    # cache features for demos and for an annotated rollout
    .venv/bin/python -m my_contributions.tools.progress_model embed bklassen3434/pick_pen_v2_20260920_124400
    .venv/bin/python -m my_contributions.tools.progress_model embed \
        bklassen3434/rollout_pen_20260925_155645 --annotations my_contributions/rollout_outcomes.json

    # train on both
    uv run --no-sync --with matplotlib python -m my_contributions.tools.progress_model train \
        bklassen3434/pick_pen_v2_20260920_124400 bklassen3434/rollout_pen_20260925_155645

    # grade a rollout
    uv run --no-sync --with matplotlib python -m my_contributions.tools.progress_model score \
        <rollout_repo_id> --all-tasks

**Held-out demonstrations:** MAE 0.016, 99.5% of within-episode frame pairs ordered correctly,
correct colour +0.97 above the wrong colour on post-grasp frames (100%).

**Real rollouts (13 episodes, verified frame by frame):** it named the pen the gripper had
closed on in 8/8 reaches, including all four where the policy grabbed the wrong pen. The arm
went to the same physical position in both halves of the session and the model's answer flipped
when the pens were swapped — so it reads colour, not position.

**Effect of adding the rollout failures**, demo metrics unchanged throughout:

| episode outcome | before | after |
| --- | --- | --- |
| lifted the pen (n=2) | 1.00 | 1.00 |
| reached, closed, failed (n=5) | 0.84 | **0.51** |
| never reached a pen (n=5) | 0.21 | **0.00** |

## Rollout harness

`tools/rollout_eval.sh` runs a policy on the SO-101 one attempt per process, so the arm is
parked while the pens are reset, and `tools/motor_retry.py` stops a single dropped Feetech
status packet from aborting an attempt.
