"""Make MuJoCo renders look like Ben's USB webcams.

MuJoCo returns a noiseless, perfectly-focused, evenly-lit image. The real cameras do
none of those things, and the differences are large enough to be a domain cue on their
own -- a policy can tell sim from real on sharpness alone. Every constant here is
measured from bklassen3434/pick_pen_v2_20260920_124400 by `python sensor.py --measure`,
not guessed.

The wrist camera matters most: at grasp range it is badly out of focus (see
out/real_wrist_at_grasp.png), while the sim's is pin-sharp. Training on a crisp wrist
view teaches the policy to rely on detail that does not exist on the robot.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter

# Measured from the real dataset (`python sensor.py` prints the targets; the gains below
# were solved to hit them). The surprise here: the sim is not too sharp, it is too FLAT.
# Raw MuJoCo renders came out at contrast 0.159 against the real camera's 0.278, and
# *less* sharp than real (0.0154 vs 0.0332) because the real frames are full of wood
# grain, cables and hardware edges that the sim scene simply does not contain. So the
# correction is mostly a contrast gain, not the blur that "sim looks too clean" suggests.
#
#   real top   sharpness 0.0332  noise 0.0357  contrast 0.2783
#   real wrist sharpness 0.0067  noise 0.0031  contrast 0.2535
#
# The wrist still gets real blur: its defocus is unmistakable in
# out/real_wrist_at_grasp.png, and the sharpness metric cannot see it because the sim's
# wrist view is a flat background with one sharp pen, which scores low for other reasons.
# `gain` is per-channel and applied about each channel's own mean, so the average colour
# is preserved and only the spread changes. A single scalar gain about mid-grey (the
# obvious approach) drove the warm wood lurid orange while still not fixing blue.
# Most of the original gap is now handled by scene content and lighting in pen_scene.xml;
# these are the residual, and they are deliberately CAPPED near 1.4. The gain that
# exactly reproduced the real blue std was 2.25, but at that strength any bright neutral
# surface (the wall) is pushed past 1.0 in blue alone and comes out lavender -- an
# artifact with no counterpart in the real frames, and therefore a domain cue of its own.
# A small std mismatch is the better trade.
TOP = dict(blur=0.40, noise=0.0210, gain=(1.05, 1.20, 1.45), vignette=0.18, warm=(0.874, 0.931, 1.034))
WRIST = dict(blur=2.20, noise=0.0020, gain=(1.08, 1.22, 1.40), vignette=0.25, warm=(0.893, 0.931, 1.015))


@lru_cache(maxsize=16)
def _vignette(shape: tuple[int, int], strength_q: int) -> np.ndarray:
    """Radial falloff mask. Cached: rebuilding the mgrid every frame cost 1.2 ms/frame
    for a mask that only changes once per episode. `strength_q` is the strength in
    thousandths so it is hashable."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    m = 1.0 - (strength_q / 1000.0) * np.clip(r / np.sqrt(2), 0, 1) ** 2
    return m[..., None].astype(np.float32)


def draw_params(cam: str, rng: np.random.Generator, jitter: float = 1.0) -> dict:
    """Sample the camera's slowly-varying settings -- ONCE PER EPISODE.

    Focus, white balance, exposure and vignetting drift over seconds on a real webcam;
    they do not resample every frame. Redrawing them per frame puts a flicker in the
    video that no real recording has, and the policy would just learn to average it out.
    Only sensor noise is per-frame, which is why it is not drawn here.

    `jitter` scales the spread: the goal is not one exact lens but the range Ben's
    cameras actually drift over between sessions.
    """
    p = TOP if cam == "top" else WRIST
    return dict(
        sigma=max(0.2, p["blur"] * (1.0 + rng.normal(0, 0.25 * jitter))),
        gain=(np.array(p["gain"]) * (1.0 + rng.normal(0, 0.03 * jitter, 3))).astype(np.float32),
        warm=(np.array(p["warm"]) * (1.0 + rng.normal(0, 0.02 * jitter, 3))).astype(np.float32),
        exposure=1.0 + rng.normal(0, 0.07 * jitter),
        vignette=p["vignette"] * (1.0 + rng.normal(0, 0.2 * jitter)),
        noise=float(p["noise"] * (1.0 + rng.normal(0, 0.3 * jitter))),
    )


def apply_params(img: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    """uint8 HxWx3 render -> uint8 HxWx3 with this episode's camera settings applied.

    Everything that does not change within an episode is computed once and cached in
    `p`; doing it per frame tripled generation time (6.0 -> 16.2 s/episode).
    """
    x = img.astype(np.float32) / 255.0
    if p["sigma"] >= 0.5:      # below this the kernel is a near no-op not worth 3.8 ms
        if p["sigma"] >= 1.6:
            # A wide blur throws away everything below ~2*sigma px anyway, so do it at
            # half resolution and scale back up: same result, ~4x cheaper. This is the
            # wrist camera's path, and the wrist is the expensive one.
            small = x[::2, ::2]
            small = gaussian_filter(small, sigma=(p["sigma"] / 2, p["sigma"] / 2, 0))
            x = np.repeat(np.repeat(small, 2, 0), 2, 1)[: x.shape[0], : x.shape[1]]
        else:
            x = gaussian_filter(x, sigma=(p["sigma"], p["sigma"], 0))

    # Per-channel spread correction about each channel's own mean, then white balance.
    # The mean is a scene-level quantity that barely moves within an episode, so it is
    # taken from the first frame and reused -- recomputing it per frame would also make
    # the correction track the arm as it sweeps across the view, which is not what a
    # camera does.
    if "mu" not in p:
        p["mu"] = x.reshape(-1, 3).mean(0)
        p["scale"] = (p["warm"] * p["exposure"]).astype(np.float32)
        p["mask"] = _vignette(x.shape[:2], int(p["vignette"] * 1000))
    x = p["mu"] + (x - p["mu"]) * p["gain"]
    x *= p["scale"]
    x *= p["mask"]
    # Per-frame sensor noise, drawn once and broadcast across the three channels: real
    # webcam noise is dominated by the luminance term, and a single-channel draw is 3x
    # cheaper than three independent ones.
    x += rng.standard_normal(x.shape[:2] + (1,), dtype=np.float32) * p["noise"]
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def degrade(img: np.ndarray, cam: str, rng: np.random.Generator, jitter: float = 1.0) -> np.ndarray:
    """Single-image convenience wrapper: draw settings and apply them in one call."""
    return apply_params(img, draw_params(cam, rng, jitter), rng)


def sharpness(img: np.ndarray) -> float:
    """Mean gradient magnitude on luma -- a scale-free proxy for focus."""
    g = img.astype(np.float32).mean(2) / 255.0
    gy, gx = np.gradient(g)
    return float(np.sqrt(gx**2 + gy**2).mean())


def noise_level(img: np.ndarray) -> float:
    """Std of the high-frequency residual: what blurring removes is noise + detail."""
    g = img.astype(np.float32).mean(2) / 255.0
    return float((g - gaussian_filter(g, 1.2)).std())


if __name__ == "__main__":
    import os

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo = "bklassen3434/pick_pen_v2_20260920_124400"
    ds = LeRobotDataset(repo, root=os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo}"))
    for cam in ("top", "wrist"):
        sh, nz, ct = [], [], []
        for ep in range(0, 60, 6):
            i = int(ds.meta.episodes["dataset_from_index"][ep]) + 150
            im = (ds[i][f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            sh.append(sharpness(im)); nz.append(noise_level(im))
            ct.append(im.astype(np.float32).std() / 255.0)
        print(f"REAL {cam:5s} sharpness {np.mean(sh):.4f}  noise {np.mean(nz):.4f}  contrast {np.mean(ct):.4f}")
