"""Does the instruction move the arm TOWARD the named pen, and far enough to reach it?

`language_sensitivity_probe.py` measures only the MAGNITUDE of the prompt's effect,
averaged over the whole 50-step action chunk. That is not sufficient to trust a
checkpoint, for two reasons this project learned the hard way:

1.  **Magnitude without direction is worthless.** `both_015000` scored 8.80 deg of
    prompt sensitivity while moving the correct way on only 8 of 16 frames -- pure
    chance. It wiggled, it did not choose.

2.  **Whole-chunk averaging understates the effect.** The early part of every chunk is
    the same swing regardless of which pen is named; the choice shows up near the
    grasp. For `contrast_010000` the whole-chunk figure was 8.91 deg but the last ten
    steps carried 12.26 deg.

And the absolute bar matters more than any comparison between checkpoints.
`contrast_010000` beat every predecessor, scored 16/16 on direction, and still went to
the same pen every time on the robot -- because 12.26 deg is only 28% of the 43.4 deg
that actually separates the two slots. A checkpoint has to move the arm *past the
midpoint* between two pens for the instruction to change which one gets picked.

What this script does, for each probed frame:
  * predicts the action chunk twice -- once with the episode's TRUE colour, once with
    the other colour -- from an identical observation with identical flow-matching noise
  * takes the mean `shoulder_pan` over the LAST `LATE_STEPS` steps of each chunk
  * asks whether the true-colour prediction lands CLOSER to where that episode's pen
    actually was (its recorded grasp pan) than the other-colour prediction does

Usage:
    cd /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1 && PYTHONPATH=. \
      uv run --no-sync python my_contributions/tools/directional_check.py \
        outputs_from_modal/v2_contrast_025000 [n_episodes] [repo_id]
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats

REPO_ID = "bklassen3434/pick_pen_v2_20260920_124400"
CACHE = "/Users/benklassen/.cache/huggingface/lerobot"

# The two taped slots in the real v2 dataset are 43.4 deg apart in shoulder_pan. To change
# WHICH pen gets picked, the prompt has to carry the arm past the midpoint, i.e. half of
# that. This is the bar `contrast_010000` failed on the robot despite looking promising.
SLOT_SEPARATION_DEG = 43.4
PASS_BAR_DEG = SLOT_SEPARATION_DEG / 2

# The pen choice is committed near the grasp, not at the start of the swing.
LATE_STEPS = 10

# Where in the approach to probe, as a fraction of the way from episode start to the grasp.
# The real operator only reaches 90% of the final pan at 73% of the way to the grasp, so
# the choice is still live through the first half -- that is where the prompt should bite.
PROBE_FRACTIONS = (0.10, 0.25, 0.40)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def grasp_rows(root: str) -> pd.DataFrame:
    """Per-episode grasp frame index and the shoulder_pan the arm actually used."""
    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files])
    rows = []
    for ep, g in df.groupby("episode_index"):
        a = np.stack(g.sort_values("frame_index")["action"].values)
        grip = a[:, 5]
        lo, hi = grip.min(), grip.max()
        gi = int(np.argmax(grip < lo + 0.3 * (hi - lo)))
        rows.append({"ep": int(ep), "grasp_frame": gi, "grasp_pan": float(a[gi, 0])})
    return pd.DataFrame(rows).set_index("ep")


def load_policy(ckpt: str, ds: LeRobotDataset):
    """Load a checkpoint and its preprocessor, handling LoRA adapters and rename maps."""
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    policy_cls = get_policy_class(cfg.type)

    if (Path(ckpt) / "adapter_config.json").is_file():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(ckpt)
        policy = policy_cls.from_pretrained(peft_config.base_model_name_or_path, config=cfg)
        policy = PeftModel.from_pretrained(policy, ckpt, config=peft_config)
    else:
        policy = policy_cls.from_pretrained(ckpt, config=cfg)
    policy = policy.to(DEVICE).eval()

    expected = [k for k in cfg.input_features if "image" in k]
    if any("images.top" in k for k in expected):
        rename_map = {}
    else:
        ours = ["observation.images.top", "observation.images.wrist"]
        rename_map = dict(zip(ours, expected))

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=ckpt,
        dataset_stats=rename_stats(ds.meta.stats, rename_map),
        preprocessor_overrides={
            "device_processor": {"device": DEVICE},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    return policy, preprocessor, cfg


def make_unnormalizer(ds: LeRobotDataset, cfg):
    """predict_action_chunk returns NORMALIZED actions; undo whichever scheme is in use."""
    st = ds.meta.stats["action"]
    if "QUANTILE" in str(cfg.normalization_mapping.get("ACTION", "MEAN_STD")).upper():
        q01 = np.asarray(st["q01"], dtype=np.float64)[:6]
        q99 = np.asarray(st["q99"], dtype=np.float64)[:6]
        return lambda x: (x + 1.0) * (q99 - q01) / 2.0 + q01
    mean = np.asarray(st["mean"], dtype=np.float64)[:6]
    std = np.asarray(st["std"], dtype=np.float64)[:6]
    return lambda x: x * std + mean


def main(ckpt: str, n_episodes: int = 16, repo_id: str = REPO_ID) -> None:
    root = f"{CACHE}/{repo_id}"
    ds = LeRobotDataset(repo_id, root=root)
    grasps = grasp_rows(root)
    tasks = list(ds.meta.tasks.index)
    if len(tasks) < 2:
        raise SystemExit(f"need >=2 tasks to test direction, {repo_id} has {tasks}")

    policy, preprocessor, cfg = load_policy(ckpt, ds)
    unnormalize = make_unnormalizer(ds, cfg)

    # Fixed noise, so the only difference between the two predictions is the prompt.
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=gen).to(DEVICE)

    eps = np.linspace(0, ds.meta.total_episodes - 1, n_episodes).astype(int)
    results, late_deltas, whole_deltas = [], [], []

    for ep in eps:
        if ep not in grasps.index:
            continue
        gi = int(grasps.loc[ep, "grasp_frame"])
        true_pan = float(grasps.loc[ep, "grasp_pan"])
        start = int(ds.meta.episodes["dataset_from_index"][ep])

        for frac in PROBE_FRACTIONS:
            idx = start + int(gi * frac)
            item = ds[idx]
            true_task = item["task"]
            other = next(t for t in tasks if t != true_task)

            base = {
                k: (v.unsqueeze(0).to(DEVICE) if isinstance(v, torch.Tensor) else v)
                for k, v in item.items()
                if isinstance(v, torch.Tensor)
            }
            pans = {}
            chunks = {}
            for prompt in (true_task, other):
                batch = dict(base)
                batch["task"] = [prompt]
                policy.reset()
                with torch.inference_mode():
                    chunk = policy.predict_action_chunk(preprocessor(batch), noise=noise)
                deg = unnormalize(chunk.squeeze(0).float().cpu().numpy()[:, :6])
                chunks[prompt] = deg
                pans[prompt] = float(deg[-LATE_STEPS:, 0].mean())

            # Did naming the TRUE colour land the arm closer to where that pen really was?
            err_true = abs(pans[true_task] - true_pan)
            err_other = abs(pans[other] - true_pan)
            improvement = err_other - err_true  # >0 means the true prompt aimed better

            results.append(
                {
                    "ep": int(ep), "frac": frac, "task": true_task,
                    "true_pan": true_pan,
                    "pan_true": pans[true_task], "pan_other": pans[other],
                    "improvement": improvement,
                    "correct": improvement > 0,
                }
            )
            a, b = chunks[true_task], chunks[other]
            late_deltas.append(np.abs(a[-LATE_STEPS:] - b[-LATE_STEPS:]).mean(axis=0))
            whole_deltas.append(np.abs(a - b).mean(axis=0))

    r = pd.DataFrame(results)
    late = np.stack(late_deltas).mean(axis=0)
    whole = np.stack(whole_deltas).mean(axis=0)
    joints = list(ds.meta.features["action"]["names"])[:6]

    print(f"\ncheckpoint : {ckpt}")
    print(f"frames     : {repo_id}  |  {len(r)} probes over {r.ep.nunique()} episodes")
    print(f"device     : {DEVICE}")

    print(f"\nmean |Δ| between prompts, DEGREES (last {LATE_STEPS} chunk steps vs whole chunk):")
    print(f"  {'joint':18s} {'late':>8s} {'whole':>8s}")
    for name, lv, wv in zip(joints, late, whole):
        print(f"  {name:18s} {lv:8.2f} {wv:8.2f}")

    pan_late = float(late[0])
    correct = int(r.correct.sum())
    print(f"\nDIRECTION: {correct}/{len(r)} probes moved toward the named pen "
          f"({100 * correct / len(r):.0f}%, chance 50%)")
    print(f"  mean improvement  {r.improvement.mean():+.2f} deg "
          f"(how much closer the true prompt aimed)")
    print("  by position in the approach:")
    for frac, g in r.groupby("frac"):
        print(f"    {int(frac * 100):3d}% of the way to grasp : "
              f"{int(g.correct.sum())}/{len(g)}  mean {g.improvement.mean():+6.2f} deg")

    print(f"\nMAGNITUDE: late-chunk shoulder_pan Δ = {pan_late:.2f} deg")
    print(f"  slots are {SLOT_SEPARATION_DEG:.1f} deg apart, so the prompt must move the arm")
    print(f"  past the {PASS_BAR_DEG:.1f} deg midpoint to change which pen is picked.")
    print(f"  -> {100 * pan_late / SLOT_SEPARATION_DEG:.0f}% of the full separation, "
          f"{100 * pan_late / PASS_BAR_DEG:.0f}% of the bar.")

    directional = correct >= 0.8 * len(r)
    if pan_late >= PASS_BAR_DEG and directional:
        print("\nVERDICT: PASS — large enough to switch pens and reliably aimed the right way.")
    elif directional:
        print("\nVERDICT: FAIL on magnitude — aims correctly but cannot reach the other pen.")
        print("         This is exactly the contrast_010000 failure: right direction, too small.")
    elif pan_late >= PASS_BAR_DEG:
        print("\nVERDICT: FAIL on direction — moves far enough but not reliably toward the target.")
    else:
        print("\nVERDICT: FAIL on both magnitude and direction.")


if __name__ == "__main__":
    main(
        sys.argv[1],
        int(sys.argv[2]) if len(sys.argv) > 2 else 16,
        sys.argv[3] if len(sys.argv) > 3 else REPO_ID,
    )
