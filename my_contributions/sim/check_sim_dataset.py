"""Audit a generated sim dataset before spending GPU hours on it.

Checks the three things that killed the earlier real datasets:

1. Colour/position decorrelation -- if a colour tends to sit in a particular place,
   the policy can solve the task from position alone and will ignore the instruction.
   This is what `slot_separation_check.py` found in v1.
2. Grasp spread -- how far apart the grasp positions actually are. v1's "three slots"
   turned out to be two, 34 deg apart with 15 deg of smear; v2 fixed this to 43.4 deg.
3. Action-space compatibility with the real dataset -- same units, same joint envelope,
   so the two can be concatenated into one training set.

    uv run --no-sync python check_sim_dataset.py --sim bklassen3434/pick_pen_sim_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
REAL_DEFAULT = "bklassen3434/pick_pen_v2_20260920_124400"


def load(repo_or_root: str) -> tuple[pd.DataFrame, dict, list[str]]:
    root = Path(repo_or_root)
    if not root.exists():
        root = Path(os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo_or_root}"))
    info = json.loads((root / "meta" / "info.json").read_text())
    df = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))]
    )
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet").index.tolist()
    return df, info, tasks


def grasp_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per episode, at the frame where the gripper closes."""
    rows = []
    for ep, g in df.groupby("episode_index"):
        a = np.stack(g.sort_values("frame_index")["action"].values)
        grip = a[:, 5]
        lo, hi = grip.min(), grip.max()
        i = int(np.argmax(grip < lo + 0.3 * (hi - lo)))
        rows.append({"episode": ep, "task_index": int(g["task_index"].iloc[0]),
                     **{n: float(a[i, k]) for k, n in enumerate(JOINTS)}})
    return pd.DataFrame(rows)


def report(name: str, df: pd.DataFrame, info: dict, tasks: list[str]) -> pd.DataFrame:
    g = grasp_rows(df)
    g["task"] = g.task_index.map(dict(enumerate(tasks)))
    n_ep = df.episode_index.nunique()
    print(f"\n=== {name} ===")
    print(f"  {n_ep} episodes, {len(df)} frames, {info['fps']} fps, tasks {tasks}")
    print(f"  episodes per task: {g.task.value_counts().to_dict()}")

    print("\n  grasp shoulder_pan by task (this is WHERE the arm went):")
    for t, sub in g.groupby("task"):
        print(f"    {t:6s} n={len(sub):3d}  mean {sub.pan.mean():+7.2f}  sd {sub.pan.std():5.2f}  "
              f"range [{sub.pan.min():+7.2f}, {sub.pan.max():+7.2f}]")
    print(f"    ALL      n={len(g):3d}  spread {g.pan.max() - g.pan.min():.1f} deg")

    # The decisive test: can you predict the target colour from where the arm went?
    # If you can, the policy never has to read the instruction.
    if g.task.nunique() >= 2:
        groups = [sub.pan.values for _, sub in g.groupby("task")]
        between = float(np.var([x.mean() for x in groups]))
        within = float(np.mean([x.var() for x in groups]))
        ratio = between / max(within, 1e-9)
        print("\n  colour<->position leakage (variance of per-colour mean pan / within-colour variance)")
        print(f"    ratio = {ratio:.4f}   ({'OK - position carries no colour info' if ratio < 0.05 else 'LEAK - colour is predictable from position'})")

        # Nearest-centroid classifier: how often does pan alone reveal the colour?
        centroids = {t: sub.pan.mean() for t, sub in g.groupby("task")}
        pred = g.pan.map(lambda p: min(centroids, key=lambda t: abs(p - centroids[t])))
        acc = float((pred == g.task).mean())
        chance = float(g.task.value_counts(normalize=True).max())
        print(f"    guessing the colour from grasp pan alone: {acc * 100:.1f}% (chance {chance * 100:.1f}%)")
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--real", default=REAL_DEFAULT)
    args = ap.parse_args()

    sdf, sinfo, stasks = load(args.sim)
    sg = report("SIM " + args.sim, sdf, sinfo, stasks)
    try:
        rdf, rinfo, rtasks = load(args.real)
        rg = report("REAL " + args.real, rdf, rinfo, rtasks)
    except FileNotFoundError:
        print("\n(real dataset not found locally; skipping comparison)")
        return

    print("\n=== co-training compatibility ===")
    print(f"  fps                {sinfo['fps']} vs {rinfo['fps']}   "
          f"{'OK' if sinfo['fps'] == rinfo['fps'] else 'MISMATCH'}")
    skeys = sorted(k for k in sinfo["features"] if k.startswith("observation.images"))
    rkeys = sorted(k for k in rinfo["features"] if k.startswith("observation.images"))
    print(f"  camera keys        {skeys}\n                     {rkeys}   {'OK' if skeys == rkeys else 'MISMATCH'}")
    print(f"  tasks              {stasks} vs {rtasks}   {'OK' if set(stasks) == set(rtasks) else 'MISMATCH'}")

    sim_a = np.stack(sdf["action"].values)
    real_a = np.stack(rdf["action"].values)
    print("\n  per-joint action range (sim vs real):")
    for i, n in enumerate(JOINTS):
        inside = sim_a[:, i].min() > real_a[:, i].min() - 5 and sim_a[:, i].max() < real_a[:, i].max() + 5
        note = "" if inside else "   <- sim explores beyond the real data"
        print(f"    {n:6s} sim[{sim_a[:, i].min():8.1f},{sim_a[:, i].max():8.1f}]  "
              f"real[{real_a[:, i].min():8.1f},{real_a[:, i].max():8.1f}]{note}")

    print(f"\n  grasp-pan spread   sim {sg.pan.max() - sg.pan.min():.1f} deg "
          f"vs real {rg.pan.max() - rg.pan.min():.1f} deg")
    print("  distinct grasp positions: sim is continuous; real is 2 fixed slots")


if __name__ == "__main__":
    main()
