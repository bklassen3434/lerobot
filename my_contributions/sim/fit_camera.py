"""Fit the sim's top camera to Ben's real webcam by eye, against a real frame.

There is no calibration for the real camera, but its pose is strongly constrained by
where the arm and pens land in the image. Rendering the sim at the real rest pose with
pens at the two measured slots and tiling candidates next to the real frame makes the
match a visual choice rather than a guess.

Usage:  python fit_camera.py            # render the candidate grid
        python fit_camera.py --pick 4   # render one candidate big, next to the real frame
"""
from __future__ import annotations

import argparse
import os
import pathlib

import imageio.v3 as iio
import mujoco
import numpy as np
from pen_env import REST_DEG, Layout, PenPickEnv

HERE = pathlib.Path(__file__).parent
REPO = "bklassen3434/pick_pen_v2_20260920_124400"

# The two taped slots, from FK of the 60 real grasp frames.
SLOT_A = (0.250, -0.026)
SLOT_B = (0.249, 0.159)

# Candidates: (distance from lookat, elevation above the table, fovy).
# The real arm base sits ~10% from the left edge, which at a 66 deg horizontal FOV
# puts the camera roughly 0.45 m from the pen area -- the v1 scene had it at 0.71 m.
# Candidates are (lookat_x, distance, elevation). In the real frame the arm base sits
# at the very LEFT EDGE, partly cut off, which means the camera aims well past it in
# +x -- aiming at the arm itself (v1) centred it instead.
CANDIDATES = [
    (lx, d, e)
    for lx in (0.26, 0.34)
    for d in (0.58, 0.70)
    for e in (38.0, 48.0)
]


def place(model, lookat_x: float, dist: float, elev_deg: float, fovy: float = 52.0) -> None:
    """Point the `top` camera at `lookat` from `dist` at `elev_deg`, looking along +y."""
    cam = model.camera("top").id
    lookat = np.array([lookat_x, 0.045, 0.01])
    e = np.deg2rad(elev_deg)
    # Camera sits at -y and above, looking toward +y and down: x_cam = +x (so world +x
    # is image-right, which is what puts the arm base on the left as in the real frame).
    offset = np.array([0.0, -dist * np.cos(e), dist * np.sin(e)])
    pos = lookat + offset
    forward = lookat - pos
    forward /= np.linalg.norm(forward)
    x_cam = np.array([1.0, 0.0, 0.0])
    y_cam = np.cross(-forward, x_cam)          # z_cam = -forward
    y_cam /= np.linalg.norm(y_cam)
    model.cam_pos[cam] = pos
    mat = np.column_stack([x_cam, y_cam, -forward]).reshape(9)
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(mat))
    model.cam_quat[cam] = q
    model.cam_fovy[cam] = fovy


def real_frame() -> np.ndarray:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO, root=os.path.expanduser(f"~/.cache/huggingface/lerobot/{REPO}"))
    i = int(ds.meta.episodes["dataset_from_index"][0])
    return (ds[i]["observation.images.top"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def render(env: PenPickEnv, lookat_x, dist, elev) -> np.ndarray:
    place(env.model, lookat_x, dist, elev)
    lay = Layout(target="pink", pen_pos={"blue": SLOT_A, "pink": SLOT_B},
                 pen_yaw={"blue": 0.0, "pink": 0.0})
    env.reset(lay, np.random.default_rng(0), REST_DEG)
    return env.render("top")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", type=int, default=None)
    args = ap.parse_args()
    env = PenPickEnv(n_pens=2)
    real = real_frame()

    if args.pick is not None:
        lx, d, e = CANDIDATES[args.pick]
        img = render(env, lx, d, e)
        iio.imwrite(HERE / "out" / "cam_pick.png", np.concatenate([real, img], 1))
        print(f"candidate {args.pick}: lookat_x={lx} dist={d} elev={e} -> out/cam_pick.png (real | sim)")
        return

    tiles = []
    for i, (lx, d, e) in enumerate(CANDIDATES):
        img = render(env, lx, d, e)
        tiles.append(img)
        print(f"  [{i}] lookat_x={lx:.2f} dist={d:.2f} elev={e:.0f}")
    rows = [np.concatenate(tiles[k:k + 2], 1) for k in range(0, len(tiles), 2)]
    grid = np.concatenate(rows, 0)
    banner = np.concatenate([real, np.zeros_like(real)], 1)
    iio.imwrite(HERE / "out" / "cam_grid.png",
                np.concatenate([banner, grid], 0)[::2, ::2])
    print("wrote out/cam_grid.png (row 0 = REAL, then candidates 0-7 left-to-right)")


if __name__ == "__main__":
    main()
