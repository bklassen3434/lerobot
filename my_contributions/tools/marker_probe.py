"""Does the green ring steer a marker-prompted checkpoint? (directional_check.py, for rings)

Same test, same bar, same numbers as directional_check.py, but the thing being swapped is
the RING, not the colour word. For each probed frame the policy predicts twice, from an
identical observation with identical flow-matching noise:
  * ring on the pen that episode actually picked (from meta/markers.json)
  * ring moved onto the OTHER pen
The prompt is the constant MARKED_TASK both times.

The rings are drawn fresh onto frames from the UNMARKED source dataset with the same
draw_marker() used to build the training set and at inference. That matters: the frames in
pick_pen_v2_marked already have the true ring burned in, so drawing the second ring there
would show the model two rings at once.

Pass bar (unchanged): late-chunk shoulder_pan Δ >= 21.7 deg (half the 43.4 deg between the
pens) AND >= 80% of probes moving toward the ringed pen.

Usage:
    cd /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1 && PYTHONPATH=. \
      uv run --no-sync python my_contributions/tools/marker_probe.py \
        outputs_from_modal/smolvla_pick_pen_v2_marked/checkpoints/010000/pretrained_model [n_episodes]
"""

import json
import sys

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from my_contributions.tools.directional_check import (
    DEVICE,
    LATE_STEPS,
    PASS_BAR_DEG,
    PROBE_FRACTIONS,
    SLOT_SEPARATION_DEG,
    grasp_rows,
    load_policy,
    make_unnormalizer,
)
from my_contributions.tools.marker_overlay import MARKED_TASK, draw_marker

CACHE = "/Users/benklassen/.cache/huggingface/lerobot"
MARKED = "bklassen3434/pick_pen_v2_marked"  # stats, markers.json, grasp pans
SOURCE = "bklassen3434/pick_pen_v2_trimmed"  # clean frames to draw rings on
TOP = "observation.images.top"


def with_ring(top: torch.Tensor, uv: list[float]) -> torch.Tensor:
    """CHW float [0,1] frame -> same frame with the ring at uv (via uint8, like training)."""
    img = (top.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return torch.from_numpy(draw_marker(img, uv)).permute(2, 0, 1).float() / 255


def main(ckpt: str, n_episodes: int = 16) -> None:
    marked = LeRobotDataset(MARKED, root=f"{CACHE}/{MARKED}")
    source = LeRobotDataset(SOURCE, root=f"{CACHE}/{SOURCE}")
    markers = {int(k): v for k, v in json.loads((marked.root / "meta" / "markers.json").read_text()).items()}
    grasps = grasp_rows(str(marked.root))

    policy, preprocessor, cfg = load_policy(ckpt, marked)
    unnormalize = make_unnormalizer(marked, cfg)

    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=gen).to(DEVICE)

    eps = np.linspace(0, marked.meta.total_episodes - 1, n_episodes).astype(int)
    results, late_deltas, whole_deltas = [], [], []

    for ep in eps:
        m = markers[int(ep)]
        gi = int(grasps.loc[ep, "grasp_frame"])
        true_pan = float(grasps.loc[ep, "grasp_pan"])
        start = int(source.meta.episodes["dataset_from_index"][m["source_ep"]])

        for frac in PROBE_FRACTIONS:
            item = source[start + int(gi * frac)]
            base = {k: v for k, v in item.items() if isinstance(v, torch.Tensor)}
            chunks, pans = {}, {}
            for which, uv in (("true", m["uv"]), ("other", m["other_uv"])):
                batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in base.items()}
                batch[TOP] = with_ring(base[TOP], uv).unsqueeze(0).to(DEVICE)
                batch["task"] = [MARKED_TASK]
                policy.reset()
                with torch.inference_mode():
                    chunk = policy.predict_action_chunk(preprocessor(batch), noise=noise)
                deg = unnormalize(chunk.squeeze(0).float().cpu().numpy()[:, :6])
                chunks[which] = deg
                pans[which] = float(deg[-LATE_STEPS:, 0].mean())

            # Did ringing the TRUE pen land the arm closer to where that pen really was?
            improvement = abs(pans["other"] - true_pan) - abs(pans["true"] - true_pan)
            results.append({"ep": int(ep), "frac": frac, "colour": m["colour"], "true_pan": true_pan,
                            "pan_true": pans["true"], "pan_other": pans["other"],
                            "improvement": improvement, "correct": improvement > 0})
            a, b = chunks["true"], chunks["other"]
            late_deltas.append(np.abs(a[-LATE_STEPS:] - b[-LATE_STEPS:]).mean(axis=0))
            whole_deltas.append(np.abs(a - b).mean(axis=0))

    r = pd.DataFrame(results)
    late = np.stack(late_deltas).mean(axis=0)
    whole = np.stack(whole_deltas).mean(axis=0)
    joints = list(marked.meta.features["action"]["names"])[:6]

    print(f"\ncheckpoint : {ckpt}")
    print(f"frames     : {SOURCE} + rings  |  {len(r)} probes over {r.ep.nunique()} episodes")
    print(f"device     : {DEVICE}")
    print(f"\nmean |Δ| between ring positions, DEGREES (last {LATE_STEPS} chunk steps vs whole chunk):")
    print(f"  {'joint':18s} {'late':>8s} {'whole':>8s}")
    for name, lv, wv in zip(joints, late, whole, strict=True):
        print(f"  {name:18s} {lv:8.2f} {wv:8.2f}")

    pan_late = float(late[0])
    correct = int(r.correct.sum())
    print(f"\nDIRECTION: {correct}/{len(r)} probes moved toward the ringed pen "
          f"({100 * correct / len(r):.0f}%, chance 50%)")
    print(f"  mean improvement  {r.improvement.mean():+.2f} deg")
    print("  by position in the approach:")
    for frac, g in r.groupby("frac"):
        print(f"    {int(frac * 100):3d}% of the way to grasp : "
              f"{int(g.correct.sum())}/{len(g)}  mean {g.improvement.mean():+6.2f} deg")

    print(f"\nMAGNITUDE: late-chunk shoulder_pan Δ = {pan_late:.2f} deg")
    print(f"  -> {100 * pan_late / SLOT_SEPARATION_DEG:.0f}% of the {SLOT_SEPARATION_DEG:.1f} deg separation, "
          f"{100 * pan_late / PASS_BAR_DEG:.0f}% of the {PASS_BAR_DEG:.1f} deg bar.")
    print("  (best language checkpoint, trim_010000: 19.42 deg)")

    directional = correct >= 0.8 * len(r)
    if pan_late >= PASS_BAR_DEG and directional:
        print("\nVERDICT: PASS — the ring retargets the arm. Worth a robot test.")
    elif directional:
        print("\nVERDICT: FAIL on magnitude — aims at the ring but not far enough to switch pens.")
    elif pan_late >= PASS_BAR_DEG:
        print("\nVERDICT: FAIL on direction — moves a lot but not reliably toward the ring.")
    else:
        print("\nVERDICT: FAIL on both magnitude and direction.")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16)
