"""Tag sim and real episodes with a domain flag and merge them into one dataset.

`MultiLeRobotDataset` raises NotImplementedError in this checkout, so co-training
means physically merging the two datasets into a single repo_id. Once merged there is
no reliable way to tell a sim frame from a real one after the fact -- episode ordering
is not a contract -- so the flag has to be stamped *before* the merge.

    uv run --no-sync python build_cotrain_dataset.py \
        --sim bklassen3434/pick_pen_sim_v1 \
        --real bklassen3434/pick_pen_v2_20260920_124400 \
        --out bklassen3434/pick_pen_cotrain_v1

Nothing is modified in place: `add_features` writes a new dataset each time, and the
source datasets are left exactly as they are. (`lerobot-edit-dataset
--operation.type=modify_tasks` does NOT behave this way -- it edits in place and
pushes, which is how the v2 dataset's task strings got rewritten on the Hub.)
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.dataset_tools import add_features, merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import dataset_to_policy_features

# 1.0 = simulated, 0.0 = recorded on the robot.
#
# The name matters. dataset_to_policy_features() only promotes keys that are image/video
# dtype or prefixed `observation.` / `action`; everything else it skips. So `is_sim`
# rides along in the batch where a sampler or an analysis script can read it, while the
# policy physically cannot. Naming it `observation.is_sim` would splice it into the
# state vector and hand the policy a one-bit shortcut for distinguishing the domains --
# which, on a task where the sim half is 3x larger and perfectly grounded, is exactly
# the kind of free signal that produces a great training curve and a useless policy.
DOMAIN_KEY = "is_sim"
DOMAIN_INFO = {"dtype": "float32", "shape": (1,), "names": None}


def resolve(repo_id: str) -> Path:
    """Filesystem path for a repo_id, or the path itself if one was given.

    Decided by how the string *looks*, not by whether it exists -- output roots do not
    exist yet, and an existence test silently buries them under the HF cache as
    ~/.cache/huggingface/lerobot/tmp/... instead of /tmp/...
    """
    if repo_id.startswith(("/", ".", "~")):
        return Path(os.path.expanduser(repo_id))
    return Path(os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo_id}"))


def tag(repo_id: str, value: float, suffix: str = "_tagged") -> LeRobotDataset:
    """Return `repo_id` with a constant DOMAIN_KEY column, adding it if absent."""
    ds = LeRobotDataset(repo_id, root=resolve(repo_id))
    if DOMAIN_KEY in ds.meta.features:
        print(f"  {repo_id}: already has {DOMAIN_KEY}; leaving as is")
        return ds

    out_repo = f"{repo_id}{suffix}"
    out_root = resolve(out_repo)
    # Tagging copies the whole dataset (~700 MB each here), so make it resumable:
    # a re-run after a downstream failure should not redo the copy.
    if (out_root / "meta" / "info.json").exists():
        done = LeRobotDataset(out_repo, root=out_root)
        if DOMAIN_KEY in done.meta.features:
            print(f"  {repo_id}: reusing existing {out_repo}")
            return done

    print(f"  {repo_id}: adding {DOMAIN_KEY}={value} -> {out_repo}")
    values = np.full((ds.meta.total_frames, 1), value, dtype=np.float32)
    return add_features(ds, {DOMAIN_KEY: (values, DOMAIN_INFO)},
                        output_dir=out_root, repo_id=out_repo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("tagging:")
    sim = tag(args.sim, 1.0)
    real = tag(args.real, 0.0)

    # aggregate_datasets() validates these deep inside the merge, after the expensive
    # copying has already happened. Check them up front with a message that says what
    # to do about it.
    if sim.meta.robot_type != real.meta.robot_type:
        raise SystemExit(
            f"robot_type differs: sim={sim.meta.robot_type!r} real={real.meta.robot_type!r}.\n"
            f"They must match for the merge. Fix the sim side -- the real value is whatever "
            f"the robot actually recorded."
        )
    if sim.meta.fps != real.meta.fps:
        raise SystemExit(f"fps differs: sim={sim.meta.fps} real={real.meta.fps}")

    print(f"\nmerging -> {args.out}")
    merged = merge_datasets([sim, real], args.out, output_dir=resolve(args.out))

    # The whole point of the naming rule: assert the policy cannot see the flag.
    policy_feats = dataset_to_policy_features(merged.meta.features)
    assert DOMAIN_KEY not in policy_feats, (
        f"{DOMAIN_KEY} leaked into the policy input features {list(policy_feats)}"
    )

    # add_features() does not write stats for the column it adds, so read the frames.
    # This is harmless for training: LeRobot only normalises features that appear in
    # the policy's input/output features, and by construction this one never does.
    frames = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(merged.root / "data" / "**" / "*.parquet"), recursive=True))]
    )
    flags = np.array([np.ravel(v)[0] for v in frames[DOMAIN_KEY]])
    n = len(flags)
    sim_frac = float(flags.mean())
    print(f"\nmerged dataset: {merged.meta.total_episodes} episodes, {n} frames")
    print(f"  sim frames  {sim_frac * n:8.0f}  ({sim_frac * 100:.1f}%)")
    print(f"  real frames {(1 - sim_frac) * n:8.0f}  ({(1 - sim_frac) * 100:.1f}%)")
    print(f"  tasks: {merged.meta.tasks.index.tolist()}")
    print(f"  policy input features: {sorted(policy_feats)}")
    print(f"  '{DOMAIN_KEY}' is present in the data but NOT a policy input -- confirmed")
    print(f"\n  root: {merged.root}")


if __name__ == "__main__":
    main()
