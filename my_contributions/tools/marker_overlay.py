"""Visual prompt for SmolVLA: draw a ring on the pen to pick, instead of naming its colour.

Every language-conditioned checkpoint mode-collapsed: the colour word never moved the arm
far enough to switch pens (best 19.4 deg late-chunk pan delta against a 21.7 deg bar). This
module moves the choice out of language and into the image. The LLM decides the colour, the
pen detector turns that into a pixel, and a ring drawn at that pixel tells SmolVLA where to go.
The prompt becomes a constant (MARKED_TASK), so there is nothing left for language to ignore.

The SAME draw_marker() is used to build the training set (mark_dataset.py) and at inference,
so the ring the policy sees on the robot is pixel-for-pixel the ring it was trained on.

The ring is placed once, from the first frame of the episode (arm at home, both pens in view),
and stays fixed for the whole episode, even after the pen is lifted. Re-detecting every frame
would be fragile (the arm hides the pen) and would make train and inference drift apart.
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

AGENT = pathlib.Path(__file__).resolve().parents[1] / "agent"
sys.path.insert(0, str(AGENT))
from perception import MAX_COLOUR_DIST, ColourModel, pixel_features  # noqa: E402

COLOURS_FILE = AGENT / "detector_colours_real.json"
MARKED_TASK = "pick up the marked pen"

# Ring style. Pure green is far from both pens, the wood and the white arm in colour space.
RING_RGB = (0, 255, 0)
RING_RADIUS_FRAC = 0.045  # of image width: 29 px at 640 wide, about a third of a pen length
RING_THICKNESS_FRAC = 0.006  # 4 px at 640 wide

# Pen-shaped blob filter, in pixels at 640x480 (scaled for other sizes).
MIN_AREA = 250
MIN_ASPECT = 4.0
MIN_LEN_PX, MAX_LEN_PX = 70, 260


def find_pen_px(img_rgb: np.ndarray, colour: str, model: ColourModel | None = None) -> tuple[float, float] | None:
    """Pixel centre (u, v) of the `colour` pen, or None if no pen-shaped blob is found.

    Pixel-only (no camera calibration needed): colour mask -> blobs -> keep long thin ones
    of pen length, clear of the frame border -> the largest wins.
    """
    model = model or ColourModel.load(COLOURS_FILE)
    h, w = img_rgb.shape[:2]
    s = w / 640
    x = pixel_features(cv2.GaussianBlur(img_rgb, (5, 5), 0), model.features)
    d = {c: model.distance(x, c) for c in model.classes}
    others = np.min([d[c] for c in model.classes if c != colour], axis=0)
    mask = ((d[colour] < MAX_COLOUR_DIST) & (d[colour] < others)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best, best_area = None, 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA * s * s:
            continue
        # Pens sit mid-table. Blobs touching the frame border are glare on the table's front
        # edge, which reads as "pink" and can out-size the real pen.
        bx, by, bw, bh = stats[i, :4]
        if bx <= 0 or by <= 0 or bx + bw >= w or by + bh >= h:
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.column_stack([xs, ys]).astype(np.float64)
        mean = pts.mean(0)
        _, sv, vt = np.linalg.svd(pts - mean, full_matrices=False)
        t = (pts - mean) @ vt[0]
        length = t.max() - t.min()
        if sv[0] / max(sv[1], 1e-6) < MIN_ASPECT or not (MIN_LEN_PX * s <= length <= MAX_LEN_PX * s):
            continue
        if area > best_area:
            # Midpoint of the two tips, not the pixel mean: the pen's clip/cap skews the mean.
            best, best_area = mean + (t.min() + t.max()) / 2 * vt[0], area
    return None if best is None else (float(best[0]), float(best[1]))


def draw_marker(img_rgb: np.ndarray, uv: tuple[float, float]) -> np.ndarray:
    """A copy of the HWC uint8 RGB image with the ring drawn at pixel uv."""
    out = np.ascontiguousarray(img_rgb).copy()
    w = out.shape[1]
    centre = (round(uv[0]), round(uv[1]))
    radius = round(RING_RADIUS_FRAC * w)
    thick = max(2, round(RING_THICKNESS_FRAC * w))
    cv2.circle(out, centre, radius, RING_RGB, thick, cv2.LINE_AA)
    return out
