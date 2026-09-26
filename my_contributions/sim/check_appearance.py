"""Acceptance test for the sim's appearance: does it pose the SAME colour problem?

Matching global pixel statistics is a proxy. What actually decides whether sim data
teaches the real task is how far each pen sits from the table it lies on -- if the sim
pens pop off a grey desk, a policy solves sim with a hue test and learns nothing.

Targets, measured off the real top camera:
    rose-gold pen vs wood   0.158   (RGB distance)
    blue pen      vs wood   0.327
    rose          vs blue   0.203
"""
from __future__ import annotations

import numpy as np
from pen_env import REST_DEG, Layout, PenPickEnv
from sensor import degrade

REAL = {"rose_vs_wood": 0.158, "blue_vs_wood": 0.327, "rose_vs_blue": 0.203}
SLOT_A, SLOT_B = (0.250, -0.026), (0.249, 0.159)


def patch(img, env, colour, rng):
    """Mean colour of a pen, found by projecting its body position into the image."""

    m, d = env.model, env.data
    cam = m.camera("top").id
    pos = d.body(f"pen_{colour}").xpos
    # world -> camera -> pixel
    cpos, cmat = d.cam_xpos[cam], d.cam_xmat[cam].reshape(3, 3)
    rel = cmat.T @ (pos - cpos)
    h, w = img.shape[:2]
    f = 0.5 * h / np.tan(np.deg2rad(m.cam_fovy[cam]) / 2)
    u = int(w / 2 + f * rel[0] / -rel[2])
    v = int(h / 2 - f * rel[1] / -rel[2])
    r = 7
    if not (r <= u < w - r and r <= v < h - r):
        return None
    return img[v - r:v + r, u - r:u + r].reshape(-1, 3).mean(0) / 255.0


def main() -> None:
    env = PenPickEnv(n_pens=2)
    rose, blue, wood = [], [], []
    for k in range(14):
        rng = np.random.default_rng(4000 + k)
        lay = Layout(target="pink", pen_pos={"blue": SLOT_A, "pink": SLOT_B},
                     pen_yaw={"blue": 0.0, "pink": 0.0})
        env.randomise_visuals(rng)
        env.reset(lay, rng, REST_DEG)
        img = degrade(env.render("top"), "top", rng)
        pr, pb = patch(img, env, "pink", rng), patch(img, env, "blue", rng)
        if pr is None or pb is None:
            continue
        h, w = img.shape[:2]
        wood.append(img[int(h * 0.62):int(h * 0.78), int(w * 0.62):int(w * 0.86)]
                    .reshape(-1, 3).mean(0) / 255.0)
        rose.append(pr); blue.append(pb)

    # Per-episode hue spread matters as much as the mean: the task signal IS the colour,
    # so a pen that renders teal in one episode and lavender in the next teaches an
    # inconsistent colour->word mapping even if the average is right.
    import colorsys
    for tag, arr in (("rose", rose), ("blue", blue)):
        hues = np.array([colorsys.rgb_to_hsv(*c)[0] * 360 for c in arr])
        # unwrap so hues near 0/360 do not report a false spread
        ref = hues[0]
        hues = ref + ((hues - ref + 180) % 360) - 180
        print(f"  {tag} hue across {len(arr)} episodes: mean {hues.mean():6.1f} deg  "
              f"sd {hues.std():4.1f}  range [{hues.min():6.1f}, {hues.max():6.1f}]")
    print()

    R, B, W = np.mean(rose, 0), np.mean(blue, 0), np.mean(wood, 0)
    got = {"rose_vs_wood": np.linalg.norm(R - W),
           "blue_vs_wood": np.linalg.norm(B - W),
           "rose_vs_blue": np.linalg.norm(R - B)}
    print(f"  {'':16s} {'SIM':>8s} {'REAL':>8s}   ratio")
    for k, target in REAL.items():
        print(f"  {k:16s} {got[k]:8.3f} {target:8.3f}   {got[k] / target:5.2f}x")
    print(f"\n  sim  rose RGB {np.round(R * 255).astype(int)}  blue {np.round(B * 255).astype(int)}"
          f"  wood {np.round(W * 255).astype(int)}")
    print("  real rose RGB [146 119 101]  blue [102 115 128]  wood [173 149 100]")


if __name__ == "__main__":
    main()
