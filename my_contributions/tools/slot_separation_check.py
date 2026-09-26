"""Check that the pen slots in a recorded dataset are actually far apart.

Reads `shoulder_pan` at the moment the gripper closes in each episode — a direct proxy for
where the target object was — and reports the clusters. Run this after the first couple of
blocks, not after the whole session: if the slots overlap, the policy can ignore the
instruction cheaply and will collapse to one habitual trajectory.

What you want to see: per-block std of a few degrees, and gaps of 25 degrees or more between
the block means. What killed the first dataset: three slots spanning ~40 degrees total, with
"centre" (std 9-14) smeared across "right".

Usage:
    cd /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1 && \
      .venv/bin/python my_contributions/tools/slot_separation_check.py <repo_id> [block_size]
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path.home() / ".cache/huggingface/lerobot"


def grasp_pan(repo_id: str) -> pd.DataFrame:
    root = Path(repo_id) if Path(repo_id).is_dir() else CACHE / repo_id
    if not root.is_dir():
        raise SystemExit(f"dataset not found: {root}")

    df = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))]
    ).reset_index(drop=True)
    actions = np.stack(df["action"].values)
    df["pan"], df["grip"] = actions[:, 0], actions[:, 5]

    tasks = pd.read_parquet(root / "meta/tasks.parquet").reset_index()
    tmap = {int(r.task_index): r.task for r in tasks.itertuples()}

    rows = []
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index").reset_index(drop=True)
        # First frame past the opening quarter where the gripper is in its most-closed 20%.
        thresh = np.percentile(g["grip"], 20)
        tail = g.iloc[len(g) // 4 :]
        closed = tail[tail["grip"] <= thresh]
        row = closed.iloc[0] if len(closed) else g.iloc[len(g) // 2]
        rows.append((int(ep), tmap[int(g["task_index"].iloc[0])], float(row["pan"])))

    return pd.DataFrame(rows, columns=["ep", "task", "pan"])


def main(repo_id: str, block: int = 15) -> None:
    r = grasp_pan(repo_id)
    r["block"] = r["ep"] // block

    print(f"\n{repo_id}: {len(r)} episodes, block size {block}\n")
    print("=== per block (each block should be one colour at one slot) ===")
    stats = r.groupby(["block", "task"])["pan"].agg(["count", "mean", "std", "min", "max"]).round(1)
    print(stats.to_string())

    means = sorted(float(m) for m in r.groupby("block")["pan"].mean().values)
    worst_std = float(r.groupby("block")["pan"].std().max())

    # Several blocks share a slot by design (each slot hosts every colour), so merge block
    # means that sit within 15 deg of each other before judging separation.
    slots: list[list[float]] = [[means[0]]]
    for m in means[1:]:
        (slots[-1] if m - slots[-1][-1] < 15 else slots.append([]) or slots[-1]).append(m)
    centres = [round(sum(s) / len(s), 1) for s in slots]
    gaps = [round(b - a, 1) for a, b in zip(centres, centres[1:])]

    print(f"\nblock means (sorted): {[round(m, 1) for m in means]}")
    print(f"slot centres: {centres}")
    print(f"gaps between adjacent slots: {gaps}")
    print(f"largest within-block std: {worst_std:.1f} deg")

    tight = worst_std < 6
    separated = all(g > 25 for g in gaps)
    print()
    if tight and separated:
        print("GOOD: slots are tight and well separated. Safe to keep recording.")
    else:
        if not tight:
            print(f"PROBLEM: within-block spread is {worst_std:.1f} deg — tape the slots down.")
        if not separated:
            print("PROBLEM: some slots are <25 deg apart — move the pens further apart and re-record.")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
