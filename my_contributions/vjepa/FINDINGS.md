# V-JEPA 2 world model of the pen scene (2026-10-04)

Three steps: (1) does the frozen encoder see what matters? (2) can a small model predict the future
from an action? (3) can it plan toward a goal picture? Step 1 is next.

## Step 1: does the frozen encoder see what matters?

**Setup.** `embed.py` ran the frozen `facebook/vjepa2-vitl-fpc64-256` encoder (ViT-L, ~300M params,
MPS, 13 clips/s) over every 3rd top-camera frame of `pick_pen_v2_20260920_124400`: 8,903 two-frame
clips, about 11 min. Frames were squashed to 256x256 rather than centre-cropped, so the pens at the
frame edges stay in. `probe.py` fits linear (ridge) probes, holding out 13 of the 60 episodes.

| question | pixels (32x24 baseline) | V-JEPA mean | V-JEPA 4x4 grid |
|---|---|---|---|
| R² shoulder_pan / lift / elbow | 0.999 / 0.998 / 0.996 | 0.991 / 0.991 / 0.985 | 0.996 / 0.996 / 0.994 |
| R² wrist_flex | 0.823 | 0.652 | 0.813 |
| R² wrist_roll | 0.981 | 0.971 | 0.984 |
| R² gripper | 0.559 | 0.514 | **0.617** |
| R² episode progress | 0.342 | **0.519** | 0.476 |
| blue pen on left? | 92.8% | **99.9%** | 98.5% (chance 54%) |
| which side is the arm going to? (every quarter) | 100% | 100% | 100% — **leak, see below** |

## What it means

1. **V-JEPA 2 sees the scene fine.** Arm pose, pen layout and phase of the task all read out
   linearly. It beats the pixel baseline on the semantic questions (pen layout, task progress, gripper)
   and ties it on big-joint pose, where 768 raw pixels of a top-down arm are already enough.
   It's a usable "eye" for a world model.
2. **Weak spots: gripper (R² ~0.6) and wrist flex.** These are small and partly hidden from the top
   camera. A world model that has to predict grasps will probably need the wrist camera too.
3. **The data leaks the instruction through the image (the important finding).** Before the arm
   moves (frames 0–90, pan drift < 0.6°), a linear probe predicts whether the episode was `blue`
   or `pink` with **100% accuracy, even from raw pixels.** The cause is recording order: v2 was
   recorded in 4 blocks (blue+left, pink+left, blue+right, pink+right).
   **What gives it away (`batch_diff.py`, `outputs/batch_diff.jpg`): the pens, not the lighting.**
   Brightness is identical across blocks (125.5 vs 125.0). But the pen you *weren't* asked for
   never got touched, so it sits on exactly the same pixels for all 15 episodes of a block (detector
   spread ±0–2 px), while the pen that got picked up was put back slightly differently each time
   (±5–11 px), and both sit in slightly different spots from block to block. A model can learn
   "pen exactly at this spot = the one that stays put = not the target". The arm's rest pose and
   cables differ a little between blocks too. Same pens, same sides, but the image still says
   which colour was asked for. So in training a policy never *needs* the word. This plausibly
   explains why the language models stalled at 19.4° on the swap-the-word probe: the probe changes
   the word but leaves the image saying "this is a blue episode". It hasn't been proven causal yet.
   (The ring passed because the ring itself is a strong image cue.)
   Episode 18 (`pink`, detector says blue-on-right) is the lone exception to the block pattern:
   either a detector miss or a real swap. Worth a look.

## For the next recording
Randomise colour *and* layout per episode within a session, and pick up and re-place **both** pens
at every reset, so neither pen's exact position predicts the instruction.

## Step 2: action-conditioned latent predictor (`world_model.py`)

A small transformer (~4M params) takes the 4x4 V-JEPA grid now, the arm state now and the next 15
actions, and predicts the grid 0.5 s later (as a change from now). 47 episodes train/val, 13 held out.
Takes about 4 min on the M3 Pro.

Held-out, "moving" pairs only (arm moves > 5° in the 0.5 s; 1,016 of 1,858):

| | copy ("nothing changes") | no-action predictor | **action predictor** |
|---|---|---|---|
| latent error (vs copy) | 1.00 | 0.75 | **0.70** |
| pan read from predicted embedding, error | 6.9° | 3.9° | **3.1°** |
| gripper read from predicted embedding, error | 3.2° | 3.1° | 3.0° |
| real future frame in nearest 5 (of ~2k held-out frames) | 9% | 44% | **64%** |

Counterfactual (same frame and arm pose, plan swapped for one heading to the other side, 308
cases): the imagined future follows the **swapped** plan 76% of the time, landing 5.9° from it
(the two futures are 17° apart). So it really is action-conditioned.

What it means:
- **It works as a world model for the arm:** it halves the error of "nothing changes", and telling it
  the plan makes it better and lets it imagine a future that didn't happen.
- **The no-action model is already good (0.75),** because the image (and the session leak) mostly
  tells it where the arm is going. The plan adds less than you'd hope.
- **It doesn't predict the grasp.** Gripper error barely beats copy, consistent with step 1's weak
  gripper read-out from the top camera.
- **Imagined futures are timid** (`outputs/world_model_examples_top.jpg`). The nearest real frame to a
  prediction often sits between "now" and the real future: right direction, too little motion.
  That's typical when training with an average-error loss.

### Adding the wrist camera (`embed.py --cam wrist`, `world_model.py --cams top wrist`)

Both cameras' grids side by side (32 tokens). Held-out, moving pairs:

| | copy | no-action | action |
|---|---|---|---|
| latent error (vs copy) | 1.00 | 0.79 | 0.76 |
| pan error | 6.8° | 4.1° | 3.4° |
| gripper error | 2.5° | 2.6° | 2.5° |
| real future in nearest 5 | 9% | 41% | 52% |

Counterfactual: follows the swapped plan 70% of the time. **On these 0.5 s scores the wrist camera
doesn't help, and still doesn't predict the gripper.** The wrist view changes a lot frame to frame,
so it's harder to predict, and in most 0.5 s windows the gripper barely moves, so this metric hardly
tests the grasp. Where the wrist camera *does* pay off is planning (below).

## Step 3: planning toward a goal picture (`plan.py`)

No policy and no reward. Given the current view and a **goal picture**, the cross-entropy method
searches for an arm motion (one joint waypoint per 0.5 s) whose imagined future, rolled out through
the world model, looks most like the goal in V-JEPA space. It starts at the moment each held-out
episode's arm begins to move and takes about 1 min per run on the M3 Pro.
- **Own goal:** the goal is this episode's real frame later on.
- **Swapped goal:** same start picture, but the goal comes from another held-out episode with the
  same pen layout and the *other* target pen. Does the plan change direction?
- **Score:** "right pen" = the plan's final shoulder_pan ends nearer the goal's pan than the other
  pen's. Coin flip = 50%. (Both pens lie the same way from the rest pose, so "moved toward the goal"
  would be meaningless. An early version of this script made that mistake and reported 100%.)

| goal 1.5 s ahead | stay still | random plan | plan (top cam) | **plan (top + wrist)** |
|---|---|---|---|---|
| own goal: right pen (13) | — | 45% | 77% | **77%** |
| own goal: pan error | 40.9° | 42.8° | 12.1° | **10.6°** |
| swapped goal: right pen (42) | — | 50% | 62% | **93%** |
| swapped goal: pan error | 39.4° | 40.4° | 17.5° | **9.1°** |

At **3 s ahead** (top + wrist) it falls apart: 46% / 57% right pen, which is a coin flip. Six chained
0.5 s predictions drift too far.

What it means:
- **Planning by imagination works over short horizons.** Show it a picture of the arm reaching the
  pink pen 1.5 s from now, and it works out the motion that gets there 93% of the time, with no
  policy trained for the task.
- **The wrist camera is what makes the swapped goals work** (62% → 93%): the goal's wrist view shows
  which pen is in front of the gripper, which is much more distinctive than the top view.
- **The planner exploits the model.** Its plans score *better* in imagination than the real demo
  actions (0.79 vs 0.88), and in `outputs/plan_examples_*.jpg` the imagined end sometimes looks like
  the goal even when the planned angle isn't there yet. It finds inputs that fool the model, so
  imagined success ≠ real success.
- **Long horizons fail** because errors compound. The fix is a model that predicts further per step
  or is trained on its own rollouts.
- Small numbers: 13 held-out episodes, one dataset, offline only. Nothing has run on the robot.

Next: run the 1.5 s planner on the real arm (top + wrist, replan every 0.5 s); train the world model
on its own multi-step rollouts; re-record leak-free data (see above) and re-run all three steps.
