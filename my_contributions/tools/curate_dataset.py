"""Curate a LeRobot dataset: find the dead time in each episode, then trim it away.

Teleop episodes are recorded on a fixed timer, so every one begins with the operator getting
ready and ends with the arm holding its final pose. On `pick_pen_v2` that is a median of 114
idle frames at the head and 8 at the tail — 29.6% of the whole dataset is the arm not moving.

Those frames are not neutral. During the idle head the correct action is "stay still" no
matter which pen was named, so a quarter of every episode actively teaches the policy that
the instruction does not matter. That is the same mode collapse this project has been
fighting, baked into the data.

Trimming keeps a short lead-in (so the policy still learns to *initiate* a motion rather than
assuming it is already underway) and a short lead-out past the grasp.

    # report only, no writing
    python my_contributions/tools/curate_dataset.py <repo_id>

    # write a trimmed copy
    python my_contributions/tools/curate_dataset.py <repo_id> --out <new_repo_id> \
        [--lead-in 15] [--lead-out 20] [--min-active 120]

Episodes whose active span is shorter than --min-active are dropped as likely failures; they
are listed in the report so you can eyeball them first.
"""

import argparse
import glob

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CACHE = "/Users/benklassen/.cache/huggingface/lerobot"
MOVE_THRESHOLD = 0.5  # summed abs joint delta per frame (deg) that counts as "moving"


def analyse(root: str) -> pd.DataFrame:
    """Per-episode motion boundaries, derived from the action stream."""
    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files]).reset_index(drop=True)

    rows = []
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index").reset_index(drop=True)
        actions = np.stack(g["action"].values)[:, :6]
        speed = np.abs(np.diff(actions, axis=0)).sum(axis=1)
        moving = np.where(speed > MOVE_THRESHOLD)[0]
        start = int(moving[0]) if len(moving) else 0
        end = int(moving[-1]) if len(moving) else len(g) - 1
        rows.append(
            {
                "ep": int(ep),
                "n": len(g),
                "start": start,
                "end": end,
                "grasp": int(np.argmin(np.stack(g["action"].values)[:, 5])),
                "active": end - start,
            }
        )
    return pd.DataFrame(rows)


def report(r: pd.DataFrame, min_active: int) -> None:
    dead = r["start"].sum() + (r["n"] - 1 - r["end"]).sum()
    total = r["n"].sum()
    print(f"episodes {len(r)} | frames {total}")
    print(f"  head idle : median {r['start'].median():.0f}  max {r['start'].max()}")
    print(f"  tail idle : median {(r['n'] - 1 - r['end']).median():.0f}  max {(r['n'] - 1 - r['end']).max()}")
    print(f"  active    : median {r['active'].median():.0f} frames")
    print(f"  DEAD      : {dead} / {total} = {100 * dead / total:.1f}% of the dataset")
    short = r[r["active"] < min_active]
    if len(short):
        print(f"\n  {len(short)} episode(s) below --min-active={min_active}, will be DROPPED:")
        print(short[["ep", "n", "active", "grasp"]].to_string(index=False))


def trim(src_repo: str, out_repo: str, r: pd.DataFrame, lead_in: int, lead_out: int, min_active: int) -> None:
    src = LeRobotDataset(src_repo, root=f"{CACHE}/{src_repo}")
    features = {k: v for k, v in src.meta.features.items() if not k.startswith(("index", "timestamp", "frame_index", "episode_index", "task_index"))}

    dst = LeRobotDataset.create(
        out_repo,
        int(src.meta.fps),
        features=features,
        root=f"{CACHE}/{out_repo}",
        robot_type=src.meta.robot_type,
        use_videos=True,
    )

    kept_frames = 0
    for row in r.itertuples():
        if row.active < min_active:
            continue
        ep_start = int(src.meta.episodes["dataset_from_index"][row.ep])
        ep_end = int(src.meta.episodes["dataset_to_index"][row.ep])
        lo = ep_start + max(0, row.start - lead_in)
        hi = min(ep_end, ep_start + min(row.end + lead_out, row.n - 1) + 1)

        for i in range(lo, hi):
            item = src[i]
            frame = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item.items() if k in features}
            # Images arrive as CHW float in [0,1]; the writer expects HWC uint8.
            for k in features:
                if "image" in k:
                    frame[k] = (np.transpose(frame[k], (1, 2, 0)) * 255).astype(np.uint8)
            frame["task"] = item["task"]
            dst.add_frame(frame)
            kept_frames += 1
        dst.save_episode()
        print(f"  ep {row.ep:3d}: kept {hi - lo:3d} / {row.n} frames", flush=True)

    print(f"\nwrote {out_repo}: {kept_frames} frames (was {r['n'].sum()})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("--out", default=None, help="write a trimmed copy under this repo id")
    p.add_argument("--lead-in", type=int, default=15)
    p.add_argument("--lead-out", type=int, default=20)
    p.add_argument("--min-active", type=int, default=120)
    a = p.parse_args()

    r = analyse(f"{CACHE}/{a.repo_id}")
    report(r, a.min_active)
    if a.out:
        print(f"\ntrimming -> {a.out} (lead_in={a.lead_in}, lead_out={a.lead_out})")
        trim(a.repo_id, a.out, r, a.lead_in, a.lead_out, a.min_active)


if __name__ == "__main__":
    main()
