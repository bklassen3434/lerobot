"""Pull table and backdrop textures straight out of the real dataset.

Inventing a wood texture is guesswork; Ben's own frames already contain the exact
surface the policy has to work against. The rose-gold pen is nearly the same hue as
this table (42.3 deg vs 41.2 deg), so the table texture is not decoration -- it is
what makes the sim's colour discrimination as hard as the real one.
"""
from __future__ import annotations

import os
import pathlib

import imageio.v3 as iio
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = "bklassen3434/pick_pen_v2_20260920_124400"
OUT = pathlib.Path(__file__).parent / "assets"

# Regions picked off out/probe_grid.png (640x480 top camera, frame 0):
#   wood     clean table, right of both pens, no cables or arm
#   backdrop wall + radiator + cable run along the top of frame
WOOD_BOX = (240, 420, 430, 620)      # y0, y1, x0, x1
BACKDROP_BOX = (0, 135, 180, 640)


def mirror_tile(patch: np.ndarray) -> np.ndarray:
    """Make a patch seamlessly tileable by mirroring it into a 2x2 block."""
    top = np.concatenate([patch, patch[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1]], axis=0)


def main() -> None:
    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{REPO}")
    ds = LeRobotDataset(REPO, root=root)

    # Average several episodes' first frames: the arm is at rest and the pens sit
    # elsewhere, so averaging suppresses noise and any stray shadow.
    frames = []
    for ep in range(0, 60, 5):
        i = int(ds.meta.episodes["dataset_from_index"][ep])
        frames.append(ds[i]["observation.images.top"].permute(1, 2, 0).numpy())
    stack = np.stack(frames)
    med = np.median(stack, axis=0)

    y0, y1, x0, x1 = WOOD_BOX
    wood = mirror_tile(med[y0:y1, x0:x1])
    iio.imwrite(OUT / "table_wood.png", (wood * 255).astype(np.uint8))

    y0, y1, x0, x1 = BACKDROP_BOX
    back = med[y0:y1, x0:x1]
    iio.imwrite(OUT / "backdrop.png", (back * 255).astype(np.uint8))

    import colorsys

    for name, patch in (("wood", wood), ("backdrop", back)):
        rgb = patch.reshape(-1, 3).mean(0)
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        print(f"{name:9s} {patch.shape}  mean RGB {(rgb * 255).round().astype(int)}  "
              f"hue {h * 360:5.1f} sat {s:.2f} val {v:.2f}")
    print("wrote", OUT / "table_wood.png", "and", OUT / "backdrop.png")


if __name__ == "__main__":
    main()
