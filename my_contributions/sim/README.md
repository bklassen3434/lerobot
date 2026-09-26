# MuJoCo digital twin of the SO-101 pen-selection scene

A simulated copy of the pen-pickup rig that generates language-grounded training
episodes at ~13 s each, in the *same action space* as
`bklassen3434/pick_pen_v2_20260920_124400`, so sim and real episodes can be
concatenated into one training set.

**Why this exists.** Every finetune so far has been less language-sensitive than the
untrained `smolvla_base`, and the contrastive-loss run that finally worked is
supervised by only 60 real episodes across 2 taped slots 43.4 deg apart. The binding
constraint is the number of distinct (instruction, target-position) pairs. A scripted
expert in sim *knows* which pen is the target, so every episode is correctly grounded
by construction, and pens can be placed anywhere rather than on two strips of tape.

## Quick start

```bash
cd /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1/my_contributions/sim

# one-time: mujoco is not in the lockfile
uv pip install -p ../../.venv/bin/python mujoco imageio

# generate 200 episodes (~45 min)
uv run --no-sync --project ../.. python generate_dataset.py \
    --episodes 200 --pens 2 --seed 11 --repo-id bklassen3434/pick_pen_sim_v1

# audit it before training on it
uv run --no-sync --project ../.. python check_sim_dataset.py --sim bklassen3434/pick_pen_sim_v1
```

## What's here

| file | purpose |
| --- | --- |
| `assets/so101_arm.xml` | SO-101 MJCF from TheRobotStudio (`so101_new_calib_camera`), plus a wrist camera and generated jaw pads |
| `assets/pen_scene.xml` | table, back wall, 3 pens, top camera, lighting |
| `pen_env.py` | env: unit conversion, domain randomisation, IK, grasp geometry |
| `scripted_pick.py` | layout sampler + the scripted expert |
| `generate_dataset.py` | runs episodes, writes a `LeRobotDataset` |
| `check_sim_dataset.py` | audits colour/position leakage and real-dataset compatibility |
| `build_cotrain_dataset.py` | tags sim/real with `is_sim` and merges them into one training set |
| `tune_grasp.py` | re-derives the grasp constants empirically |
| `add_jaw_pads.py` | regenerates the fingertip collision pads from the jaw meshes |

## Domain tagging (sim vs real)

`MultiLeRobotDataset` raises `NotImplementedError` in this checkout, so co-training
means physically merging sim and real into one `repo_id` — after which nothing
distinguishes a sim frame from a real one. So every frame carries an **`is_sim`**
column (1.0 sim, 0.0 real).

```bash
uv run --no-sync --project ../.. python build_cotrain_dataset.py \
    --sim bklassen3434/pick_pen_sim_v1 \
    --real bklassen3434/pick_pen_v2_20260920_124400 \
    --out bklassen3434/pick_pen_cotrain_v1
```

**The name is load-bearing.** `dataset_to_policy_features()` only promotes keys that are
image/video dtype or prefixed `observation.` / `action`; everything else it skips. So
`is_sim` reaches the batch — where a sampler, a weighter, or an analysis script can read
it — while the policy physically cannot. Naming it `observation.is_sim` would splice it
into the state vector and hand the policy a one-bit shortcut for telling the domains
apart, which on a set where the sim half is 3x larger and perfectly grounded is exactly
the sort of free signal that yields a great loss curve and a useless policy. The merge
script asserts the flag is absent from the policy features.

Nothing is edited in place — `add_features` writes a new dataset and leaves the sources
alone. (`lerobot-edit-dataset --operation.type=modify_tasks` does *not* behave this way:
it edits in place and pushes.)

The merged set: **260 episodes / 75,763 frames, 64.8% sim / 35.2% real.**

Three quirks found while wiring this up:

- `robot_type` must match exactly or `aggregate_datasets()` refuses the merge -- and it
  checks *after* copying both datasets. The real data says `so_follower` (the robot class
  in this checkout is `SOFollower`), not `so101_follower`. `build_cotrain_dataset.py` now
  pre-checks `robot_type` and `fps` up front, and tagging is resumable so a downstream
  failure does not force a re-copy of ~1.4 GB.

- `add_features` does not write `meta/stats.json` entries for the column it adds. Harmless
  — `lerobot_train.py` passes `dataset.meta.stats` wholesale and the normaliser indexes it
  by *policy* features — but it means you cannot read the sim fraction out of stats.
- `lerobot_train.py` already supports per-sample loss weighting via a `SampleWeighter`,
  which receives the whole batch and so can read `is_sim` directly. **But** it calls
  `policy.forward(batch, reduction="none")`, and `contrastive_language_loss.py` short-circuits
  to the unpatched forward whenever `reduction != "mean"` — so enabling a weighter would
  silently disable the contrastive penalty, with no error. Combining the two needs the
  patch to return per-sample losses first.

## Measured results

- **Scripted-pick success: 81%** over the full 200-episode run (247 attempts). Failures
  are discarded, so the dataset contains only correct demonstrations.
- **Grasp `shoulder_pan` spans 114 deg** of continuous positions across the 200 written
  episodes, against the real v2 dataset's two fixed slots 43.4 deg apart.
- **Colour/position leakage 0.0008**, and the colour is guessable from grasp pan alone
  only 51.0% of the time (chance 50%) -- as clean as the real Latin square (0.0001, 50.0%).
- Exactly 100 blue / 100 pink episodes, 49,066 frames, 679 MB, 47 min to generate.
- `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` all stay inside the real
  dataset's observed ranges. `shoulder_pan` deliberately goes beyond them.
- ~13 s per episode including rendering both 640x480 cameras and AV1 encoding.

## Things that were not obvious, and cost the most time

**The MJCF joint convention is identical to LeRobot's calibrated degrees.** `qpos =
deg2rad(action)` with no offsets or sign flips, for the `so101_new_calib` model. Two
independent confirmations: the model's joint limits coincide with the real dataset's
extremes to within a degree (elbow limit 96.8 vs observed max 97.2), and forward
kinematics of the 60 real grasp frames lands the tool on the table at exactly two
tight clusters — the taped slots — at x=0.250/y=-0.026 and x=0.249/y=+0.159.

**...except the gripper, which is not degrees at all.** LeRobot gives the five arm
joints in degrees but the gripper in `RANGE_0_100` percent. Passing percent through
`deg2rad` puts the jaws ~20 mm apart when the robot reports "closed". `to_rad` /
`from_rad` in `pen_env.py` handle the conversion. The sim's zero still doesn't line up
with the real robot's (sim closes on a pen at 9%, the real robot reports ~2%), so
`generate_dataset.py` rescales the channel on the way out.

**MuJoCo collides meshes by convex hull, which fills in the jaws' concave inner
faces.** The hull faces are 16-20 mm apart even fully closed, and only come within a
pen's diameter ~17 mm behind the fingertips — which would mean driving the tips
through the table. `add_jaw_pads.py` adds explicit box pads at the tips, sized from the
measured mesh geometry. Without them the pen is only ever wedged, and slips out during
the lift.

**The jaws are not symmetric about the tool axis.** Only one jaw moves, so the point
where the pads meet at a 10 mm gap sits **8.5 mm off-axis**. Aiming IK at the tool axis
misses the pen by that much every time. This is the `z` component of
`GRASP_SITE_OFFSET`, and it is the single fix that took scripted-pick success from 0%
to 80%. Everything before it — tuning the approach depth, the closing command, the
pad thickness — was tuning around a systematic 8.5 mm miss.

**The approach tilt points outward, not inward.** The real grasp frames have approach
axes whose horizontal component points *away* from the base. Getting this sign wrong
gives IK solutions that are valid but put the wrist in a posture the real robot never
uses.

**Solving each waypoint with independent IK puts the arm on a different manifold.**
Unconstrained IK reached the same hover points with elbow at -73 deg and wrist_flex
pinned at its +95 deg limit — the real robot never goes below +7 deg elbow or above
+49 deg wrist_flex. Two fixes: `IK_LIMITS` pins elbow/wrist to the real branch, and
`plan()` now makes exactly **one** IK call (the grasp) with the approach and lift as
joint-space offsets from it. Those offsets are measured from the real episodes, which
show the operator is already in the grasp posture 1.5 s before the grasp — only 9 deg
higher on `shoulder_lift` — and rolls the wrist ~49 deg while lifting. There is no
10-cm hover with the tool held at the grasp orientation; that was an invention.

**`shoulder_pan` must NOT be constrained to the real dataset's range.** The real data
spans only [-89.5, +12.7] because two strips of tape were the only places a pen ever
sat. Pinning pan to that shrank the workspace to a sliver and rejected 40% of layouts.
Generating pens at *new* pan angles is the most valuable thing this sim does.

**A diagnostic trap:** `mujoco.mj_forward` must be re-run after writing to
`model.cam_pos` / `cam_quat`, or `update_scene` renders from the stale pose. Four
camera placements in a row produced byte-identical images before I noticed.

## Known sim2real gaps

1. **Top camera extrinsics are a guess.** The real webcam's pose is unknown, so the
   nominal pose is a stand-in and `randomise_visuals` jitters position, orientation and
   FOV every episode. This is the largest visual gap.
2. **Gripper channel zero differs** (rescaled on write, see above). The gripper is
   effectively binary for this task, so only its two levels matter.
3. **`shoulder_pan` goes beyond the real range**, by design. If you want sim layouts to
   match the real slot geometry exactly, narrow `Y_RANGE` in `scripted_pick.py`.
4. **The rest pose jitters more than the real robot's.** Start poses are `REST_DEG` plus
   noise with sigma 2 deg on elbow, where the real robot's frame-0 elbow has sd 0.2 deg.
   That pushes sim `elbow_flex` to 102.6 vs the real max of 97.2. It acts as a mild
   augmentation rather than a problem, but it is why the audit flags elbow as
   out-of-range alongside the deliberate `shoulder_pan` excursion.
5. **Rendering is clean MuJoCo**, not webcam pixels — no motion blur, rolling shutter,
   sensor noise or JPEG artefacts. Co-training with the real episodes rather than
   pretraining on sim alone is the intended mitigation.

## Suggested next experiment

Co-train on sim + real rather than sim alone, roughly 3:1 sim:real, with the existing
contrastive language loss and frozen VLM, then run the usual gate:
`language_sensitivity_probe.py` **and** the directional check. The question this sim is
built to answer is whether more distinct (instruction, position) pairs raises
directional accuracy and prompt sensitivity above `contrast_010000`'s 16/16 and
12.3 deg late-chunk pan swing.

---

## v2: matching Ben's actual scene (2026-09-25)

The v1 sim looked obviously synthetic, and measuring *why* turned up something more
serious than cosmetics.

### The sim was posing an easier problem than the robot faces

Measured off the real top camera (`out/probe_grid.png`, frame 0):

| | RGB | hue | sat |
|---|---|---|---|
| wood table | (173,149,100) | 40.5 deg | 0.42 |
| "pink" pen | (146,119,101) | **24.0 deg** | 0.31 |
| "blue" pen | (102,115,128) | 210.0 deg | 0.20 |

Ben's "pink" pen is a **metallic rose-gold pen on a golden-brown desk** -- 0.158 away
from the table in RGB, against 0.327 for the blue one. The v1 sim rendered it as hot
magenta (sat 0.59, val 1.00) on a grey checkerboard. A policy could solve v1 sim with a
crude hue test and learn nothing that transfers, which is consistent with the measured
3.2x drop in prompt sensitivity between sim and real frames.

`check_appearance.py` is the acceptance test, and it targets those separations rather
than "does it look nice":

```
                        SIM     REAL   ratio
  rose_vs_wood        0.155    0.158    0.98x
  blue_vs_wood        0.354    0.327    1.08x
  rose_vs_blue        0.237    0.203    1.17x
```

### What changed

- **Textures come from the dataset**, not from imagination (`extract_textures.py`):
  the table is a mirror-tiled crop of Ben's own desk, the backdrop is his wall and
  cable run.
- **The arm is white**, matching the real 3D-printed SO-101. It was the published
  MJCF's yellow, and it fills a large share of both views.
- **Camera fitted to the real frames** (`fit_camera.py`): the real view puts the arm
  base at the very left edge, which pins the camera to aim ~0.34 m out in +x from
  0.58 m at 48 deg. v1 aimed at the arm from 0.71 m and centred it.
- **Scene clutter** -- cables, USB hub, the dark void under the desk. Not decoration:
  the real frames' dynamic range comes from having a white wall and near-black cables
  in the same image.
- **`sensor.py`** reproduces the webcam: per-episode focus/white-balance/exposure/
  vignette (drawn once per episode -- redrawing per frame adds a flicker no recording
  has), per-frame luminance noise, and the wrist camera's heavy defocus.

### Three things that were counter-intuitive

1. **The sim was not too sharp -- it was too FLAT.** Raw renders measured contrast
   0.159 against the real camera's 0.278, and *lower* sharpness (0.0154 vs 0.0332),
   because real frames are full of wood grain, cables and hardware edges. "Sim looks
   too clean, add blur" would have made it worse.
2. **Matching a summary statistic is not matching an image.** Driving a spotlight hard
   reproduced the real per-channel std exactly -- by burning a hotspot into the table.
   Likewise the gain that reproduced the real blue std (2.25) turned the white wall
   lavender. Both were reverted in favour of content and softer light, accepting a
   small std mismatch.
3. **Hue is unstable on a near-neutral object.** The blue pen is only sat 0.20, so an
   independent per-channel jitter of 0.04 rendered it teal in one episode and lavender
   in the next -- an inconsistent colour->word mapping in a task where colour IS the
   signal. Pen jitter is now mostly luminance, and `check_appearance.py` reports
   per-episode hue spread (blue sd 24.2 deg -> 10.8 deg) so this cannot regress silently.
