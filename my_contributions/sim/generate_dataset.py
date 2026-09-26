"""Generate a LeRobotDataset of scripted pen picks from the MuJoCo twin.

Writes exactly the feature set of the real dataset
(bklassen3434/pick_pen_v2_20260920_124400) so the two can be concatenated and
co-trained: same fps, same camera keys, same 6-D action/state in LeRobot units.

    uv run --no-sync python generate_dataset.py --episodes 300 --repo-id me/pick_pen_sim

Failed picks are dropped, so the dataset contains only correct demonstrations. The
scripted expert knows which pen is the target, so every episode is correctly
language-grounded by construction -- which is the whole reason for generating it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pen_env import COLOURS, FPS, REST_DEG, PenPickEnv  # noqa: E402
from scripted_pick import GRIPPER_CLOSED, GRIPPER_OPEN, rollout, sample_layout  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

# The sim gripper and the real one do not share a zero. Ben's calibration gives the
# gripper 1460 ticks of travel where the MJCF declares 110 degrees, so "closed on a
# pen" reads ~9% in sim and ~2% on the robot, and "open" reads 28% vs ~23%. The
# gripper is effectively binary for this task, so map the sim's two levels onto the
# real ones rather than leaving a channel whose marginal distribution disagrees.
REAL_GRIP_CLOSED, REAL_GRIP_OPEN = 2.0, 23.0

# Must match the real dataset exactly: aggregate_datasets() refuses to merge datasets
# whose robot_type differs, and the recorded data says "so_follower" (the robot class
# in this checkout is SOFollower), not "so101_follower".
ROBOT_TYPE = "so_follower"


def remap_gripper(a: np.ndarray) -> np.ndarray:
    """Rescale the sim gripper channel onto the real dataset's range, in place-safe."""
    out = np.asarray(a, np.float32).copy()
    scale = (REAL_GRIP_OPEN - REAL_GRIP_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED)
    out[..., 5] = REAL_GRIP_CLOSED + (out[..., 5] - GRIPPER_CLOSED) * scale
    return out


FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        ],
    },
    "observation.images.top": {
        "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"],
    },
    "observation.images.wrist": {
        "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"],
    },
    # Domain tag: 1.0 = simulated, 0.0 = recorded on the robot. Deliberately NOT named
    # `observation.is_sim` -- dataset_to_policy_features() only promotes keys that are
    # video/image or prefixed `observation.`/`action`, so this name stays metadata the
    # sampler and analysis can see but the policy physically cannot read. Naming it
    # `observation.is_sim` would concatenate it into the state vector and hand the
    # policy a free shortcut for telling the domains apart.
    "is_sim": {"dtype": "float32", "shape": (1,), "names": None},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100, help="successful episodes to write")
    ap.add_argument("--repo-id", default="bklassen3434/pick_pen_sim")
    ap.add_argument("--root", default=None)
    ap.add_argument("--pens", type=int, default=2, choices=(2, 3),
                    help="2 = blue/pink, matching the real v2 dataset; 3 adds grey")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", type=float, default=1.0, help="domain-randomisation strength")
    ap.add_argument("--max-attempts", type=int, default=0, help="0 = 4x episodes")
    args = ap.parse_args()

    colours = COLOURS[: args.pens]
    env = PenPickEnv(n_pens=args.pens)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=FPS, features=FEATURES, root=args.root,
        robot_type=ROBOT_TYPE, use_videos=True,
    )

    max_attempts = args.max_attempts or args.episodes * 4
    written = attempts = 0
    per_colour = dict.fromkeys(colours, 0)
    t0 = time.time()

    while written < args.episodes and attempts < max_attempts:
        # Round-robin the target so the colours stay exactly balanced.
        target = colours[written % len(colours)]
        rng = np.random.default_rng(args.seed * 1_000_003 + attempts)
        attempts += 1

        layout = sample_layout(rng, n_pens=args.pens, target=target)
        env.randomise_visuals(rng, strength=args.dr)
        start = REST_DEG + rng.normal(0, [3.0, 2.0, 2.0, 2.0, 3.0, 1.0])
        res, info = rollout(env, layout, rng, start, render=True)
        if res is None or not info["success"]:
            continue

        action = remap_gripper(res["action"])
        state = remap_gripper(res["state"])
        for t in range(len(action)):
            ds.add_frame({
                "action": action[t],
                "observation.state": state[t],
                "observation.images.top": res["frames"]["top"][t],
                "observation.images.wrist": res["frames"]["wrist"][t],
                "is_sim": np.ones(1, np.float32),
                "task": target,  # one-word labels, as the real v2 dataset now uses
            })
        ds.save_episode()
        written += 1
        per_colour[target] += 1
        if written % 10 == 0 or written == args.episodes:
            el = time.time() - t0
            print(f"  {written}/{args.episodes} episodes "
                  f"({attempts} attempts, {written / attempts * 100:.0f}% success) "
                  f"{el / 60:.1f} min, {el / written:.1f} s/ep", flush=True)

    ds.finalize()
    print(f"\nwrote {written} episodes to {ds.root}")
    print(f"  per colour: {per_colour}")
    print(f"  attempts {attempts}, scripted-pick success {written / max(attempts, 1) * 100:.0f}%")
    print(f"  total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
