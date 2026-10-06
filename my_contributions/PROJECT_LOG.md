# Project Log — every dataset, every trained model, and why it exists

Written 2026-09-25. This is the inventory + the story. For the *how*, see
[README.md](./README.md); for the efficiency milestones, see the `M*_FINDINGS.md` files.

---

## There are really two projects here

**Track A — "train a foundation model efficiently" (done).** Profiling, mixed precision,
gradient accumulation, LoRA, DDP/FSDP. Ran on rented Modal GPUs against `lerobot/pusht` and
`smolvla_base`. Milestones M1–M4, all finished, written up in `M1`–`M4_FINDINGS.md`. This
track produced *benchmarks*, not robot behaviour.

**Track B — "pick the pen I name" (in progress, the hard one).** Teach an SO-101 arm to pick
the blue / pink / grey pen *on command*. Every dataset and every checkpoint below except the
M1–M4 benchmark runs belongs to this track. The whole track is one long fight against a
single failure mode:

> **The policy learns the motion but ignores the words.** Behaviour cloning is only ever
> asked "did you predict the demonstrated actions?", never "did you predict them *because of*
> the instruction". When two tasks share a scene and differ by one word, ignoring the word and
> averaging the two targets is the cheaper solution — and gradient descent takes it.

The measured symptom: change the prompt from "blue" to "pink" and `shoulder_pan` (the joint
that chooses *which* pen) moves **0.4–1.9°**. Actually switching pens needs **43.4°**. That
held across seven different training configs — and then a custom contrastive loss broke it
(10.2°, and the arm moves toward the *named* pen 16/16 times). **The current best checkpoint
is `outputs_from_modal/contrast_010000`, and it has never been put on the robot.**

---

## Datasets

All recorded with `lerobot-record` on the SO-101 at 30 fps, two cameras. "eps" = episodes.

### Track B lineage (the pen-selection work)

| Dataset | eps / frames | Task strings | Cameras | What it is / why |
|---|---|---|---|---|
| `so101_pick_object_20260713_221513` | 90 / 41,070 | `pick up the keys` · `pick up the pen` · `pick up the sanitizer` | overhead, wrist | **First multi-task attempt (July).** Three *different objects*, so "which object" was confounded with "what it looks like". |
| `pick_pen_20260919_165739` ("v1") | 90 / 40,296 | `Pick up the blue pen` · `…pink pen` · `…grey pen` | top, wrist | **The real v1.** Three identical pens, differing only in colour — so language is the *only* cue. 30 eps per colour. Killed by geometry: the three taped slots spanned only ~40° of `shoulder_pan` and the "centre" slot smeared into "right" (std 9–14°), so one habitual trajectory could satisfy all three prompts. Found with `tools/slot_separation_check.py`. |
| `pick_pen_v2_20260920_124400` ("v2") | 60 / 26,697 | initially `Pick up the blue pen` / `…pink pen`; **relabelled 2026-09-21 to bare `blue` / `pink`** | top, wrist | **The workhorse.** Dropped to two pens so the slots could be pushed far apart and unambiguous. 30 eps each. Every model from `lr5e5` onward trains on this. The relabelling to single words is itself an experiment — see the `words` run. |
| `pick_pen_v2_trimmed` | 60 / 20,311 | `blue` / `pink` | top, wrist | **v2 with the dead time cut out (newest, 2026-09-25).** Episodes were recorded on a fixed 15 s timer, so a median of 114 idle frames at the head and 8 at the tail — 24% of all frames — were the arm sitting still. During the idle head the correct action is "don't move" *regardless of which pen was named*, so a quarter of the data was actively teaching the policy that the instruction doesn't matter. Built by `tools/curate_dataset.py`. Trained on — see the 2026-09-26 entry. |

### Earlier single-task work (pre-language, May–Aug)

| Dataset | eps / frames | Task | Note |
|---|---|---|---|
| `pen_lift_20260523_123133` | 25 / 13,304 | `Pick up pen and lift it` | First real recording session. |
| `pen_lift_c270_20260523_162743` | 25 / 13,755 | same | Re-shot with the C270 webcam; trained the first ACT policy. |
| `pen_lift_20260825_223702` | 25 / 12,232 | same | August re-shoot. |
| `rollout_act_pen_lift_c270_v1_20260524_125347` | 1 / 2,710 | — | Not training data: a **recorded rollout** of the ACT policy, for playback/inspection. |
| `glasses_case_lift_*` (6 repos, 1 ep each) | 1 / 600 | glasses case | Throwaway smoke tests from the very first sessions. |
| `_smoke_delete_me_*`, `so101_pick_object_2207*/2214*/2215*`, `pen_lift_2026052*_1230*`, `pen_lift_c270_*_1358*` | 0–1 eps | — | Aborted / empty recordings. Safe to delete. |

**Local only vs on the Hub:** everything with real episode counts in the two tables above is
pushed to `huggingface.co/datasets/bklassen3434/…` **except** the empty/smoke repos, which
only exist in `~/.cache/huggingface/lerobot/`.

---

## Trained models

Local copies in `outputs_from_modal/<tag>_<step>/`; Hub copies at
`huggingface.co/bklassen3434/…`. Every Track-B run starts from pretrained
`lerobot/smolvla_base` (450 M params) — nothing is trained from scratch.

### Track B: the pen-selection ladder, in the order it was tried

The scoreboard column is **`shoulder_pan` Δ in degrees when only the prompt changes**, measured
by `tools/language_sensitivity_probe.py`. **43.4°** is what switching pens actually requires.
The reference point that matters: *untrained* `lerobot/smolvla_base` scores **8.8°**, so for
most of this project every fine-tune made language sensitivity **worse than doing nothing**.

| # | Local dirs | Hub repo | Dataset | The knob being turned | pan Δ | Verdict |
|---|---|---|---|---|---|---|
| — | (base model) | `lerobot/smolvla_base` | — | untrained control | **8.8°** | the bar to beat |
| 0 | — | `smolvla_so101` | `so101_pick_object_…221513` | First language-conditioned attempt, 3 objects, 20k steps | — | moved, but "which object" was never cleanly testable |
| 1 | `smolvla_pick_pen_10000`, `_20000` | `smolvla_pick_pen` | **v1** (90 eps, 3 pens) | Full fine-tune, lr 1e-4, vision encoder unfrozen, 20k steps | 1.9° | smooth pick — but the *same* pick every time, right-hand pen regardless of prompt |
| 2 | `lr5e5_*` | `…_v2_lr5e5` | v2 | **Halve the LR to 5e-5** — maybe full-strength fine-tuning on 60 eps wipes out the base model's grounding | 0.4° | worst of all; and *falling* with training |
| 3 | `lr1e4_010000` | `…_v2_lr1e4` | v2 | Control: same but lr 1e-4 on v2 | ~0.4° | no |
| 4 | `frozen_*` | `…_v2_frozen` | v2 | **Freeze the whole VLM** (`freeze_vision_encoder` + `train_expert_only`) so the language pathway *physically cannot* be overwritten | 1.8° | no → the problem is **not** catastrophic forgetting. Freezing prevents forgetting; it does not force *use*. |
| 5 | `lora_*` | `…_v2_lora` | v2 | **LoRA** r=16 α=32 on the action expert, lr 1e-4 — 743K trainable params (0.16%) | 0.55° | no |
| 6 | `lora3e4_*` | `…_v2_lora3e4` | v2 | Same LoRA at lr 3e-4 | 0.95° | no → closes the whole hyperparameter/regularisation branch |
| 7 | `pi05_002000/004000` | `…_v2_pi05` | v2 | **Swap the architecture**: `lerobot/pi05_base`, bf16 + gradient checkpointing, bs 32 | 0.05° | worst measured. And *untrained* pi05_base also scores 0.046° — it is prompt-insensitive on this embodiment before we touch it. Closes the "use a bigger VLA" branch. |
| 8 | `words_*` | `…_v2_words` | v2, **relabelled** | **Change the data, not the model**: shorten the prompt to just `blue` / `pink`, so the differing token is 1-in-3 instead of 1-in-10 | 1.9° @10k, 2.2° @25k | still far too small — but it **tripled** the same config on phrases and was the first run in seven where sensitivity *rose* with training instead of decaying. First evidence the conditioning **signal**, not the model, is the binding constraint. |
| 9 | `contrast_002500/005000/010000` | `…_v2_contrastive` | v2 | **Change the loss**: `tools/contrastive_language_loss.py` — a second forward pass with instructions *shuffled across the batch*, penalised for still predicting the demo under someone else's instruction. λ=1.0, margin 2.0, ~2× compute/step | **10.2° / 9.1° / 8.9°** | ✅ **the breakthrough.** 4–5× the same config without the penalty, and above the untrained base for the first time. Imitation quality unharmed (5.02° vs 5.00° per-joint error). Training log tells the story: `bc=0.058` vs `wrong=4.52` — 78× separation, and the penalty decayed to zero because the margin was satisfied everywhere. |
| 10 | `sd_002500` → `sd_025000` | `…_v2_statedrop` | v2 | **Remove the shortcut**: `tools/state_dropout.py` hides the joint readings on 50% of samples (masked *after* normalisation, so 0 = average pose) | 0.8° / 1.0° / 3.1° / 2.7° / 3.2° | partial. Beats plain words by ~60% and *improved* imitation accuracy (4.90°), but plateaued at ~3° — the steep 5k→10k climb was the end of the ramp, not the start. |
| 11 | `both_005000/010000/015000` | `…_v2_both` | v2 | **#9 + #10 together**, 15k steps | 8.8° @15k | did **not** beat the penalty alone, and see the directional check below — dropout slightly *dilutes* the penalty. Best imitation accuracy of any run though (4.18°), so dropout is a mild regulariser, not a second lever. |

### The check that settled it

The Δ probe measures only the **magnitude** of the prompt's effect, not its **direction** — a
policy that swings the *wrong* way scores just as well. So: for each frame, ask with the
episode's true colour vs the other colour, and check whether the true-colour prediction lands
*closer* to where that pen actually is.

| Checkpoint | Moves toward the named pen | Signed shift |
|---|---|---|
| `contrast_010000` | **16/16 (100%)** | +10.7° |
| `both_015000` | 8/16 (50% = chance) | +2.2° |
| `words_025000` | 10/16 (62%) | +0.7° |

`both_015000`'s 8.8° of "sensitivity" is directionally **random** — dropout made it wiggle, not
choose. `contrast_010000` actually chooses. **Always run the directional check, not just the Δ
probe, before trusting a checkpoint.**

Shared settings for #2–#11 unless stated: batch 64, A100-80GB on Modal, `save_freq=2500`,
LR schedule decay matched to the run length, cameras renamed `top→camera1` / `wrist→camera2`
(SmolVLA's baked-in slots).

### Track A / earlier, for completeness

| Model | Policy | Dataset | Note |
|---|---|---|---|
| `act_pen_lift_c270_20260523_162743` | ACT | `pen_lift_c270_…162743` | 100k steps, bs 8 — the first policy that ever worked on this rig |
| `act_pen_lift_25ep_v1` | ACT | `pen_lift_20260825_223702` | 20k steps, bs 8 |
| M1–M4 benchmark runs | SmolVLA | `lerobot/pusht` | Throwaway checkpoints; the artifact is the numbers, in `M*_FINDINGS.md`. Profiler trace: `outputs_from_modal/smolvla_m1_trace.json` |

---

## The strategy, in one page

**Phase 1 — get *anything* working (May–Aug).** Single task, ACT, 25 episodes. Proved the rig,
the calibration, and the recording loop.

**Phase 2 — make it language-conditioned (July–Sept).** Move to SmolVLA, a vision-language-
action foundation model, and record multiple tasks in one dataset. Fail three times in
instructive ways:
- *three different objects* → "which one" confounded with appearance;
- *three identical pens too close together* → one trajectory satisfies every prompt (caught by
  `slot_separation_check.py`, which reads `shoulder_pan` at the moment the gripper closes);
- *two well-separated pens* → clean data, and the collapse **still** happens.

**Phase 3 — attack the collapse from every side (Sept).** The ladder above is deliberately
ordered from cheap-and-conventional to expensive-and-novel, and each rung falsifies a
hypothesis rather than just "trying stuff":

| Hypothesis | Rung | Verdict |
|---|---|---|
| The LR is destroying the base model's language grounding | #2, #3 | **No** |
| Fine-tuning is overwriting the VLM | #4 (freeze it entirely) | **No** — freezing prevents forgetting, doesn't force use |
| Too many trainable params for 60 episodes | #5, #6 (LoRA) | **No** — closes the tuning branch |
| SmolVLA specifically is the problem | #7 (π0.5) | **No** — pi05_base is *already* prompt-blind on this embodiment |
| The instruction is diluted by boilerplate words | #8 | **Partly** — 3× gain, first metric that *rose* with training |
| **Nothing in the loss rewards using the instruction** | #9 (contrastive) | **YES. This was it.** 10.2°, 100% directionally correct |
| The joint state is a shortcut making vision+language unnecessary | #10 (state dropout) | Partly — plateaus at 3°, but improves imitation accuracy |
| Both at once | #11 | Doesn't compound; dropout dilutes the penalty's direction |
| A quarter of the data teaches "the words don't matter" | `pick_pen_v2_trimmed` | Built, **not yet trained** |

The shape of the answer: rungs #2–#7 all tried to fix the *model*, and all failed. Rungs #8–#10
changed the *supervision* — what the prompt looks like, what the loss asks for, what
information the policy is allowed to lean on — and that is where the movement came from. The
one-line takeaway: **behaviour cloning has no term that rewards using the instruction, so you
have to add one.**

**Phase 4 — build the measurement tools, because eyeballing the arm doesn't scale.** Two
were needed:
- `language_sensitivity_probe.py` — run the checkpoint on real frames, once per prompt, with
  *identical* observations and *identical* flow-matching noise. Unnormalises back to degrees
  and reports per-joint Δ, scoring `shoulder_pan` against the 43.4° of slot separation ("% of
  the distance needed"). ~0 = the instruction is ignored. Five minutes on the laptop instead of
  an afternoon of robot trials. **This is the number that diagnosed the whole project.** Two
  things it took a correction to get right: only the first 6 of SmolVLA's 32 action dims are
  real joints (the rest are padding that never moves and dilutes the average), and the output
  must be unnormalised before comparing against degrees. Always probe untrained
  `lerobot/smolvla_base` as the reference — for most of this project it beat every fine-tune.
- `progress_model.py` + `progress_labels.py` — a language-conditioned progress model, 0 = nothing
  has happened, 1 = **the named** pen is in the gripper. Frozen SigLIP tower + small MLP head;
  labels derived automatically from the action stream (arm starts moving → gripper closes →
  pen is up), never from a clock, never by hand; the *wrong* colour is trained as a
  zero-progress negative, so "always grab the same pen" scores as a failure rather than a
  success. Held-out v2: MAE 0.031, 99.4% of within-episode frame pairs ordered correctly,
  correct colour scores +0.97 over the wrong one. Zero-shot on the v1 dataset (different
  session, different layout, three pens, none of it in training): all six probe episodes score
  1.00 for the right colour, ≤0.44 for the wrong one.

**Phase 5 — sim, the newest thread (2026-09-25, in progress).** `my_contributions/sim/` is a
MuJoCo digital twin of the pen scene that emits episodes in the *exact same action space* as
the real dataset (`[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`
in LeRobot degrees), so sim and real episodes can be co-trained in one `LeRobotDataset`. The
point: a scripted expert *knows* which pen it was told to pick, so every sim episode is
correctly language-grounded by construction and there can be as many as you want — which is
exactly the supervision 60 real episodes were too few to provide. Convention validated by
forward-kinematics against the 60 real grasp frames. Includes a fix for a real MuJoCo gotcha
(`add_jaw_pads.py`: meshes collide by convex hull, which fills in the concave jaw faces, so
the gripper can never actually touch a pen without explicit pad geoms).

---

## Tooling inventory (`my_contributions/tools/`)

| File | What it does |
|---|---|
| `language_sensitivity_probe.py` | Does the checkpoint's output change when the prompt changes? The core diagnostic. |
| `slot_separation_check.py` | Are the pen slots far enough apart that the policy *can't* cheat? Run after the first few recording blocks, not at the end. |
| `curate_dataset.py` | Finds and trims the dead time at the head/tail of every episode. Produced `pick_pen_v2_trimmed`. |
| `progress_labels.py` | Event-derived per-frame progress labels in [0,1], read out of the action stream. |
| `progress_model.py` | The language-conditioned success detector (`embed` → `train` → `score`). |
| `contrastive_language_loss.py` | Monkeypatch: makes ignoring the instruction expensive. |
| `state_dropout.py` | Monkeypatch: hides the joint state half the time. |
| `pi05_compat.py` | Shim so `lerobot/pi05_base` loads in this checkout (processor-step name drift). |
| `camera_check.py` | Align the live camera against dataset frames before recording, so a new session matches the old one. |

Plus `my_contributions/modal/` — `modal_m1..m4.py` for Track A, `modal_train_pick_pen.py` for
every Track-B run (it takes `--lora`, `--freeze-vlm`, `--contrastive`, `--state-dropout`,
`--policy pi05`, `--dataset`, `--tag`, and checkpoints to a persistent Modal volume so a dead
run resumes instead of restarting).

Pipeline changes live in the LeRobot source, tagged `[learning-project]`:
`grep -rn "\[learning-project\]" src/`.

---

## Where it stands, and the loose ends

**The single most important open item: `outputs_from_modal/contrast_010000` has never been run
on the robot.** Everything above is offline measurement. Eval prompts must be the bare words
`blue` / `pink` (that is what it was trained on), with the same rename map
(`top→camera1`, `wrist→camera2`).

Other open threads:
- **`pick_pen_v2_trimmed` exists but nothing has been trained on it.** It removes the 24% of
  frames that explicitly teach "the instruction is irrelevant", which is the one data-side
  lever that hasn't been pulled. Obvious next run: `--contrastive 1.0` on the trimmed data.
- **The progress model has never seen a real rollout.** It validates beautifully on training
  data and zero-shot on v1, but no `pick_pen` rollout dataset has ever been recorded, so the
  test that matters — "does it separate a good checkpoint from the collapsed one" — is
  outstanding and needs robot time.
- **The sim thread stops at a scripted pick** (`sim/out/*.png`). No sim dataset has been
  emitted or co-trained yet.
- Trimmed-dataset video files are **asymmetric across cameras** (`top` rolled into 3 mp4s,
  `wrist` into 1). Decoding through `LeRobotDataset` is verified correct and training is
  unaffected, but opening the raw mp4s side by side looks catastrophically desynced. Worth
  making rebuilds symmetric.
- `lr1e4_010000` and `lr5e5_010000` are stored one level deeper (`<dir>/pretrained_model/`)
  than every other checkpoint dir — mind that when passing `--policy.path`.
- ~10 empty/smoke dataset repos are still in the local cache and can be deleted.

---

## 2026-09-25 — `contrast_010000` FAILED ON THE ROBOT, and the probe did not predict it

Ben put `contrast_010000` on the SO-101. **It goes to the same location every time,
regardless of whether the prompt is `blue` or `pink`.** Mode collapse, same as every
earlier checkpoint.

This is the most important result in the project so far, because that checkpoint was the
best thing the offline metrics had ever produced:

| metric | contrast_010000 | what it needs to be |
|---|---|---|
| whole-chunk `shoulder_pan` Δ | 8.91 deg | — |
| late-chunk pan Δ (near grasp) | 12.26 deg | — |
| directional check | **16/16 (100%)** | — |
| signed shift toward the named pen | +10.70 deg | — |
| **slot separation actually required** | — | **43.4 deg** |

### The probe was not wrong. It was over-interpreted.

12.26 deg is **28% of the 43.4 deg needed to switch pens**. The policy nudges the arm
toward the correct pen — the 16/16 directional score is real and reproducible — but it
never travels far enough to land on a different one. A 12 deg bias against a 43 deg
requirement puts the gripper on the same pen every time. That is exactly what the robot
did.

Every ranking in this log compared checkpoints against *each other* and against the
untrained base (8.8 deg), and treated "above baseline and directionally correct" as
promising. None of them compared against the **absolute bar**, which is the only thing
that decides whether a pen gets picked.

### Corrected acceptance criterion

A checkpoint is worth robot time only if late-chunk `shoulder_pan` Δ exceeds roughly
**half the slot separation (~21.7 deg)**, so the prompt can carry the gripper past the
midpoint between two pens, and ideally approaches the full 43.4 deg. On that bar:

| config | late-chunk pan Δ | % of 43.4 | verdict |
|---|---|---|---|
| contrast 10k | 12.26 | 28% | **fails on robot (confirmed)** |
| both 15k | 8.91 | 21% | fails, and directionally random (8/16) |
| statedrop 25k | 4.70 | 11% | fails |
| plain words 25k | 2.70 | 6% | fails |

**No checkpoint ever produced has cleared the bar.** The project has been ranking
degrees of failure.

### What this does NOT mean

The probe is still a valid *screen* — it is cheap, it runs on the laptop, and a
checkpoint that scores 0.5 deg is certainly dead. What changed is the pass mark: it is
an absolute 21.7+ deg, not "better than the last run".

### Consequences for work in flight

- `cotrain_plain` (run A) finished flat at 0.64-0.75 deg across all checkpoints. Dead by
  any standard.
- `cotrain_contrast` (run B) must be judged against 21.7 deg, not against 8.91.
- The sim v2 appearance rebuild is still worth finishing — the sim/real probe gap (3.2x)
  is a separate, independently measured problem — but nobody should expect appearance
  alone to close a 4x shortfall in prompt authority.


---

## 2026-09-26 — trimmed data trained: best numbers yet, still under the bar

`v2_trim_contrast` = the exact winning config (frozen VLM, contrastive λ=1.0, word labels,
10k steps) retrained on `pick_pen_v2_trimmed`. Only the data changed, so this isolates what
curation alone is worth. Checkpoints local: `outputs_from_modal/trim_005000`, `trim_010000`.

Both models judged on **identical eval frames** (from the trimmed set), each unnormalizing
with **its own training stats** — getting either of those wrong invalidates the comparison:

| metric | `contrast_010000` (untrimmed) | `trim_010000` (trimmed) |
|---|---|---|
| whole-chunk pan Δ | 8.91 deg | **10.4–11.5 deg** |
| late-chunk pan Δ (near grasp) | 12.26 → 13.38 deg | **19.42 deg** |
| imitation error vs ground truth | 20.82 deg | **5.87 deg** |
| directional check | **16/16** | 13/16 |

**Curation is the single biggest lever found so far.** +45% late-chunk prompt authority and a
3.5× better imitation error, from deleting 24% of the frames.

The imitation number also corrects an earlier mistake in this log: `contrast_010000` was
previously credited with ~5.0 deg imitation error, but that was measured on frames drawn from
its own idle head — it was being graded on standing still. On *moving* frames it scores 20.82.

### Against the corrected bar

19.42 deg is **45% of the 43.4 deg slot separation**, i.e. ~90% of the way to the 21.7 deg
acceptance criterion. Closest anything has come. Still below it, so by the rule adopted on
2026-09-25 it should **not** get robot time yet — the expected outcome is the same mode
collapse `contrast_010000` showed, and we already know what that looks like.

### Known cosmetic defect in `pick_pen_v2_trimmed`

The rebuild gave the two cameras different mp4 file splits (`top` → 3 files, `wrist` → 1).
Frame totals match (20,311 each) and per-episode metadata is internally consistent per camera;
pixel-matching trimmed frames back to source at 5/35/65/95% of episodes 0/20/40 **and** at the
file-rollover boundaries (31/32/33/55/56/59) shows both cameras resolving to the same source
frame everywhere. **Training is unaffected.** But opening the raw mp4s side by side looks
wildly desynced, because `top/file-000.mp4` holds episodes 0–31 while `wrist/file-000.mp4`
holds all 60. `curate_dataset.py` should force symmetric file rollover before the next rebuild.

---

## Where we are

- **Nine training configurations.** Exactly one intervention has ever moved prompt authority
  materially: the **contrastive counterfactual loss** (`tools/contrastive_language_loss.py`).
  Everything else — bigger model (π0.5), LoRA, learning rate, state dropout, more steps —
  was flat or worse.
- **Two levers now proven, and they stack differently than expected:** the contrastive loss
  buys prompt authority; **data curation** buys both authority *and* execution quality. State
  dropout is a mild accuracy regulariser, not a language lever (it made the model wiggle at
  chance direction, 8/16).
- **Nothing has cleared the 21.7 deg bar.** Best = `trim_010000` at 19.42 deg.
- **One robot test has been run** (`contrast_010000`, 12.26 deg): mode collapse, as the
  corrected criterion predicts.

## Where we're going

Ranked by expected value per hour:

1. **Push the contrastive loss harder on the trimmed data.** λ=2.0 (and/or margin 3.0), 15k
   steps, on `pick_pen_v2_trimmed`. λ=1.0 got 19.42; the penalty had decayed to zero by the end
   of training (`penalty=0.0000` in the logs), meaning the margin was satisfied and the term
   stopped pushing. A larger margin keeps it pushing. This is one command and the cheapest
   shot at clearing the bar.
2. **More episodes.** 30 per instruction is the thinnest thing in the project. The rig is still
   taped at +10 deg / −33.5 deg, so 30 more per colour is ~2 h of teleop and the one lever
   nobody has actually pulled.
3. **Finish the curation toolkit** — grasp-failure detection (did the gripper stay closed after
   closing?) and phase labelling (train on the approach segment only, where the colour choice
   happens). Both are cheap and compound with (1) and (2).
4. **Only then**: robot time, and only for a checkpoint above 21.7 deg.

---

## 2026-10-02 — new strategy: marker-prompted SmolVLA (LLM picks colour, ring marks the pen)

Language never got past 19.4 deg of the 21.7 deg bar, so the colour choice moves out of
language and into the image. Claude picks the colour, the pen detector finds it in the
top camera, a green ring is drawn there for the whole episode, and SmolVLA runs with the
constant prompt `pick up the marked pen`.

- `tools/marker_overlay.py`: `find_pen_px()` (pixel-only detector, no calibration) and
  `draw_marker()`, shared by dataset building and inference so the ring is identical in both.
- `tools/mark_dataset.py`: built **`bklassen3434/pick_pen_v2_marked`** from `pick_pen_v2_trimmed`
  (60 episodes, 20,311 frames, 1 task, wrist camera and actions unchanged). Both pens were found
  in 60/60 episodes; contact sheet `pick_pen_v2_marked_check.jpg` was checked by eye, and every ring
  is on the named pen. `meta/markers.json` stores the ring and the *other* pen's pixel per episode,
  so the offline probe can move the ring onto the other pen.
- Each colour appears in both table positions across the recording sessions, so the ring is the only
  cue for which side to go to.
- Detector trap: glare on the table's front edge reads as "pink" and out-sized the real pen in 8
  episodes. Fixed by ignoring blobs that touch the frame border.

Training: `smolvla_pick_pen_v2_marked` on Modal (frozen VLM, 10k steps, batch 64, no contrastive
loss), launched 2026-10-03; final checkpoint to be pushed to `bklassen3434/smolvla_pick_pen_v2_marked`.

Probe: `tools/marker_probe.py` — directional_check.py with the RING swapped instead of the colour word
(rings drawn fresh on clean `pick_pen_v2_trimmed` frames, since the marked frames already have one).
Same bar: late-chunk pan Δ >= 21.7 deg and >= 80% toward the ringed pen. Noise floor: `trim_010000`,
which never saw rings, scores 3.4 deg just from the image changing (6 probes).

### Result: the ring PASSES the probe — first checkpoint ever to clear the bar

`smolvla_pick_pen_v2_marked` (Hub: `bklassen3434/smolvla_pick_pen_v2_marked`, final 10k checkpoint),
`marker_probe.py`, 48 probes over 16 episodes:

| | late-chunk pan Δ | direction | verdict |
|---|---|---|---|
| ring-unaware `trim_010000` (noise floor) | 3.4 deg | — | — |
| best language model `trim_010000` (colour word) | 19.42 deg | 13/16 | fail |
| **marked 5k** | **29.98 deg** | **48/48** | **PASS** |
| **marked 10k** | **30.03 deg** | **48/48** | **PASS** |

138% of the 21.7 deg bar, 69% of the full 43.4 deg separation, mean improvement +27.7 deg, 100%
directional at every point in the approach. Same data, same model, same training config as the best
language run; only the prompt channel changed (word -> ring). Already saturated at 5k steps.

Caveat: the probe has never been validated by a PASS on the robot (the only robot test was a fail it
predicted). Next: a robot runner (detector -> ring -> SmolVLA), Ben to OK the first hardware run.

### Robot runner built (2026-10-03, NOT yet run on hardware)

`tools/marker_rollout.sh "<instruction>" <attempts> [repo]`: per attempt, `marker_aim.py` (Claude
haiku turns the instruction into blue/pink, the top camera is read, both pens are found, and the ring
preview is opened) -> Ben confirms the ring -> `lerobot-rollout` (sentry, 15 s, recorded) with
`marker_shim.py` patching `SOFollower.get_observation` to draw the ring on every top frame. All
rollout_eval.sh behaviour (motor_retry, return to start, resume) is kept.

Dry runs, no hardware:
- Claude: "pink pen"/"blue one"/"rose-gold pen"/"write in blue ink" were all correct; "banana" -> refused.
- Full robot code path (patched get_observation -> build_dataset_frame -> SyncInferenceEngine, the
  exact calls lerobot-rollout makes), from the PARKED frame: the end of the first chunk matches the
  human demo at frame 50 within 2-5 deg (ep 0/20/40/59: -30.7/-41.8/-38.4/-6.0 vs -34.6/-43.6/-40.4/-10.6),
  and moving the ring to the other pen shifts it 22-34 deg the other way.

### Camera alignment tool (2026-10-03)

Measured: the top camera moved **<1 px / <0.15 deg / <0.3% zoom** across all 60 training episodes
(ORB + RANSAC against ep 59). So the marker model has seen exactly ONE camera view; putting the camera
back matters. (Pen placement varied between sessions; the camera did not.)

`tools/camera_align.py`: live top feed vs the ep-59 reference (edges / blend views), with live shift /
rotation / zoom numbers, nudge hints and pen-detector status. Verified on synthetic offsets (15 px shift,
8 px, 2 deg roll, 5% zoom, mixed): every measurement was correct to within 0.1 px / 0.03 deg, with the right hint direction.
The "GOOD" bar (4 px, 0.4 deg, 1%) is a cautious guess; the policy's real tolerance is unmeasured.

### Camera tolerance + first live aim (2026-10-03)

- `marker_aim.py` bug: the USB top camera sends BLACK frames for ~1-2 s after opening, and the
  first version used frame 15. It now waits for lit frames and then settles. (This was not a
  permissions problem, even inside the Claude app's terminal.)
- Live frame after Ben re-aligned: shift (+1.7, -4.2) px, zoom 99.8%, but **rolled -2.3 deg**.
- Tolerance measured by rotating probe frames (ring moved with the pen): 2.3 deg -> 27.2 deg, 47/48;
  5 deg -> 27.1 deg, 48/48 (unrotated: 30.0, 48/48). **Pen CHOICE is robust to a few degrees of roll.**
  This does not cover grasp precision, which the probe doesn't measure.

## 2026-10-03 — FIRST ROBOT RUN of the marker model: right pen 3/3

`marker_rollout.sh`, dataset `bklassen3434/rollout_marker_20261003_153409` (local). Ring placed by
Claude + detector each time; pens in the usual two spots.

| attempt | asked | pen | pan when gripper closed | training grasp pan for that pen |
|---|---|---|---|---|
| 1 | pink | upper | -34.7 | -35.0 (range -56..-31) |
| 2 | blue | lower | +6.4 | +9.9 (range +7..+12) |
| 3 | pink | upper | -33.5 | -35.0 |

**Pen choice 3/3, in BOTH directions.** That is the first time in the project the arm has gone to
different pens on command. (Every language checkpoint went to the same place every time.) The gripper closed to ~5%
at the pen (air closes to ~0.75%, a pen stalls it at ~2-4%), then reopened to ~15% and the arm went home,
like the demos (which put the pen back). Whether the pen was actually LIFTED needs Ben's eyes or the
wrist video. Attempt 2 dipped twice (lift -22 -> +5 between 6 s and 8 s): a possible regrasp.

The "Record loop 1.7 Hz" warnings are the ~0.5 s MPS inference stall at each 50-step chunk boundary,
not a slow loop: about 22 Hz overall (328-358 frames in 15 s).

---

## 2026-10-04 — V-JEPA 2 encoder probe (world-model step 1) + recording-order leak

`vjepa/embed.py` + `vjepa/probe.py`; full write-up in `vjepa/FINDINGS.md`. Frozen V-JEPA 2 ViT-L
embeddings of the v2 top camera linearly recover arm pose (R² ≥0.98 big joints), pen layout (99.9%)
and task progress, beating a raw-pixel baseline on the semantic ones. Gripper is weak (R² ~0.6).
**Leak:** from idle frames before the arm moves, even raw pixels predict the `blue`/`pink`
instruction with 100% accuracy, because v2 was recorded in 4 colour x layout blocks. A plausible
reason the language runs stalled. Fix for future recordings: interleave colour and layout.

**Correction + step 2 (same day).** The leak is not lighting: `vjepa/batch_diff.py` shows the
*untouched* pen sits on identical pixels for a whole block while the picked pen moves a few px each
reset. Fix: re-place both pens every reset and randomise colour/layout per episode.
`vjepa/world_model.py` (action-conditioned latent predictor, 0.5 s ahead) beats copy (0.70 latent
error on moving frames; real future in top-5 retrieval 64% vs 9%) and follows a swapped plan 76% of
the time. It doesn't predict the grasp (gripper ≈ copy). Details in `vjepa/FINDINGS.md`.

**Wrist camera + planning (2026-10-04/05).** Adding the wrist camera doesn't improve 0.5 s prediction,
but it makes goal-picture planning work: `vjepa/plan.py` (CEM search through the world model, no
policy) reaches the right pen 77% (own goal) / **93%** (goal picture from another episode) of the time
1.5 s ahead, vs 50% random. At 3 s it's a coin flip (rollout drift). The planner exploits the model
(beats the demo actions in imagination), so these are offline numbers only. Details in `vjepa/FINDINGS.md`.
