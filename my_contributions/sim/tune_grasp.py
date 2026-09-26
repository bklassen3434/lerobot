"""Re-derive the grasp constants empirically. Run this after changing the scene.

Three sweeps, each answering one question:

  --workspace   where can a pen go and still be graspable within the real robot's
                joint branch?  -> X_RANGE / Y_RANGE in scripted_pick.py
  --gripper     what closing command actually holds a pen?  -> GRIPPER_CLOSED
  --pads        where do the fingertip pads meet, in the tool frame?
                -> GRASP_SITE_OFFSET in pen_env.py

The numbers baked into the modules came from these sweeps; none of them were guessed.

    uv run --no-sync python tune_grasp.py --pads --gripper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import pen_env  # noqa: E402
import scripted_pick  # noqa: E402
from pen_env import REST_DEG, Layout, PenPickEnv, to_rad  # noqa: E402
from scripted_pick import GRASP_SEED, rollout, sample_layout  # noqa: E402

PEN_DIAMETER_MM = 10.0


def sweep_pads(env: PenPickEnv) -> None:
    """Where the two fingertip pads sit, in the tool-site frame, vs gripper command.

    The pen is held where the pads meet at its own diameter. Because only one jaw
    moves, that point is NOT on the site's axis -- which is what GRASP_SITE_OFFSET's
    z component corrects for.
    """
    m, d = env.model, env.data
    fp, mp = m.geom("fixed_jaw_pad").id, m.geom("moving_jaw_pad").id
    q = np.array([8.4, 7.0, 35.7, 31.0, -84.9, 0.0])
    print("\n--- pad geometry (tool-site frame, mm) ---")
    print(" grip%   fixed_z  moving_z   gap   midpoint_z")
    rows = []
    for gp in np.arange(0.0, 20.1, 1.0):
        q[5] = gp
        d.qpos[:6] = to_rad(q)
        mujoco.mj_forward(m, d)
        s = d.site("gripperframe")
        R = s.xmat.reshape(3, 3)
        f = (d.geom_xpos[fp] - s.xpos) @ R * 1000
        v = (d.geom_xpos[mp] - s.xpos) @ R * 1000
        gap = v[2] - f[2] - 3.5  # minus the two pad half-thicknesses
        rows.append((gp, f[2], v[2], gap, (f[2] + v[2]) / 2))
        print(f" {gp:5.1f}  {f[2]:8.2f}  {v[2]:8.2f}  {gap:6.2f}  {(f[2] + v[2]) / 2:9.2f}")
    arr = np.array(rows)
    i = int(np.argmin(np.abs(arr[:, 3] - PEN_DIAMETER_MM)))
    print(f"\n  pads reach the pen's {PEN_DIAMETER_MM:.0f} mm diameter at gripper={arr[i, 0]:.1f}%, "
          f"midpoint z={arr[i, 4]:.2f} mm")
    print(f"  -> GRASP_SITE_OFFSET z should be {arr[i, 4] / 1000:.4f}  "
          f"(currently {pen_env.GRASP_SITE_OFFSET[2]:.4f})")
    print(f"  -> GRIPPER_CLOSED must be BELOW {arr[i, 0]:.1f}% or the jaws never touch the pen")


def sweep_workspace(env: PenPickEnv) -> None:
    env.reset(Layout(target="blue", pen_pos={"blue": (0.25, 0.0)}, pen_yaw={"blue": 0.0}),
              np.random.default_rng(0))
    ys = np.arange(-0.22, 0.28, 0.03)
    xs = np.arange(0.16, 0.35, 0.02)
    print("\n--- workspace: fraction of approach tilts that solve within the joint branch ---")
    print("  x\\y  " + "".join(f"{y:+6.2f}" for y in ys))
    good = []
    for x in xs:
        row = f" {x:.2f} "
        for y in ys:
            p = np.array([x, y, 0.0026])
            yaw = np.arctan2(y, x)
            n = 0
            for tilt in (16, 22, 28, 32):
                seed = GRASP_SEED.copy()
                seed[0] = -np.degrees(yaw)
                _, err = env.ik_grasp(p, yaw, tilt, seed)
                n += err < 0.004
            row += f"{n / 4:6.2f}"
            if n == 4:
                good.append((x, y))
        print(row)
    g = np.array(good)
    print(f"\n  fully solvable: x[{g[:, 0].min():.2f},{g[:, 0].max():.2f}] "
          f"y[{g[:, 1].min():+.2f},{g[:, 1].max():+.2f}]  "
          f"(currently X_RANGE={scripted_pick.X_RANGE} Y_RANGE={scripted_pick.Y_RANGE})")


def sweep_gripper(env: PenPickEnv, n: int) -> None:
    print(f"\n--- grasp success vs closing command ({n} episodes each) ---")
    original = scripted_pick.GRIPPER_CLOSED
    for gc in (0.0, 3.0, 6.0, 9.0, 12.0, 15.0):
        scripted_pick.GRIPPER_CLOSED = gc
        ok = ikf = 0
        for i in range(n):
            rng = np.random.default_rng(9000 + i)
            lay = sample_layout(rng, n_pens=2)
            _, info = rollout(env, lay, rng, REST_DEG, render=False)
            ok += bool(info.get("success"))
            ikf += info.get("fail") == "ik"
        print(f"  GRIPPER_CLOSED={gc:5.1f}%  success {ok / n * 100:4.0f}%  ik-reject {ikf}")
    scripted_pick.GRIPPER_CLOSED = original


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pads", action="store_true")
    ap.add_argument("--workspace", action="store_true")
    ap.add_argument("--gripper", action="store_true")
    ap.add_argument("-n", type=int, default=40)
    args = ap.parse_args()
    if not (args.pads or args.workspace or args.gripper):
        args.pads = args.workspace = args.gripper = True

    env = PenPickEnv()
    if args.pads:
        sweep_pads(env)
    if args.workspace:
        sweep_workspace(env)
    if args.gripper:
        sweep_gripper(env, args.n)


if __name__ == "__main__":
    main()
