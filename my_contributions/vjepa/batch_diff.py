"""What differs between recording batches that "should" look identical?

v2 was recorded in 4 blocks of 15: blue+blue-left, pink+blue-left, blue+blue-right, pink+blue-right.
Blocks 1 vs 2 (and 3 vs 4) have the same pens on the same sides, so any visible difference between
them is a session fingerprint the policy could use instead of the instruction.

For each block, average the idle frames (0-60, before the arm moves) and write a contact sheet:
  row 1: block "blue" mean | block "pink" mean | |difference| x5 (heat map)    (blue pen on left)
  row 2: same, blue pen on right
Also prints, per block: brightness, colour balance, and where the detector found each pen.

    cd /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1 && \
        .venv/bin/python my_contributions/vjepa/batch_diff.py
"""

import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from embed import CACHE, OUT_DIR, find_pen_px, iter_frames, load_meta  # noqa: E402

REPO = "bklassen3434/pick_pen_v2_20260920_124400"
IDLE = 60
BLOCKS = {"blue, blue-left": range(0, 15), "pink, blue-left": range(15, 30),
          "blue, blue-right": range(30, 45), "pink, blue-right": range(45, 60)}


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main() -> None:
    root = CACHE / REPO
    eps, _ = load_meta(root)
    sums = {ep: np.zeros((480, 640, 3)) for ep in eps.episode_index}
    first = {}
    for ep, idx, img in iter_frames(root, eps):
        if idx <= IDLE:
            sums[ep] += img
            if idx == 0:
                first[ep] = img
    ep_mean = {ep: s / (IDLE + 1) for ep, s in sums.items()}

    block_mean = {}
    print(f"{'block':<18} {'bright':>7} {'R':>6} {'G':>6} {'B':>6}   blue pen (px)      pink pen (px)")
    for name, rng in BLOCKS.items():
        m = np.mean([ep_mean[e] for e in rng], axis=0)
        block_mean[name] = m
        blue = np.array([find_pen_px(first[e], "blue") or (np.nan, np.nan) for e in rng])
        pink = np.array([find_pen_px(first[e], "pink") or (np.nan, np.nan) for e in rng])
        r, g, b = m.reshape(-1, 3).mean(0)
        print(
            f"{name:<18} {m.mean():7.1f} {r:6.1f} {g:6.1f} {b:6.1f}   "
            f"({np.nanmean(blue[:, 0]):4.0f},{np.nanmean(blue[:, 1]):4.0f}) ±{np.nanstd(blue, 0).mean():3.0f}   "
            f"({np.nanmean(pink[:, 0]):4.0f},{np.nanmean(pink[:, 1]):4.0f}) ±{np.nanstd(pink, 0).mean():3.0f}"
        )

    # within-block spread, for scale: how different are two halves of the SAME block?
    rows = []
    for side, (a, b) in {"blue-left": ("blue, blue-left", "pink, blue-left"),
                         "blue-right": ("blue, blue-right", "pink, blue-right")}.items():
        diff = np.abs(block_mean[a] - block_mean[b]).mean(2)
        rng_a = list(BLOCKS[a])
        same = np.abs(np.mean([ep_mean[e] for e in rng_a[::2]], 0) - np.mean([ep_mean[e] for e in rng_a[1::2]], 0)).mean(2)
        print(f"{side}: mean |blue block - pink block| = {diff.mean():.2f} grey levels; "
              f"two halves of the same block = {same.mean():.2f}")
        heat = cv2.applyColorMap(np.clip(diff * 5, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        rows.append(np.hstack([label(block_mean[a].astype(np.uint8), f"asked BLUE ({side})"),
                               label(block_mean[b].astype(np.uint8), f"asked PINK ({side})"),
                               label(heat, "difference x5")]))
    out = OUT_DIR / "batch_diff.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
