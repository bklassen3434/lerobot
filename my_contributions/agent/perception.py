"""Fast pen detector for the top camera: pixels -> "blue pen at (x, y), angle a".

This is the "eyes" behind the find() tool. Claude's own vision can say WHICH object is
which, but reading exact positions off a grid is slow and only good to ~5 mm. This
detector finds every pen in a few milliseconds, to ~1-2 mm, and hands Claude a
labelled picture to sanity-check.

How it works:
  1. Colour: each pixel's chroma (colour with brightness removed, Lab a/b channels) is
     compared with learned colour models for "blue pen", "pink pen" and "table". Ignoring
     brightness makes it robust to shadows and highlights.
  2. Shape: matching pixels are grouped into blobs; only long, thin blobs count as pens.
  3. Geometry: the blob's two ends are traced back through the calibrated camera onto
     the table plane (at pen-centre height). Centre = midpoint, angle = direction.

The colour models are fitted by fit_detector.py from sim renders whose pixels are
labelled by MuJoCo's segmentation view. On the real rig you'd fit them from a handful
of hand-labelled frames, or swap this whole module for a learned detector (YOLO/OWLv2).
"""

from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MODEL_FILE = HERE / "detector_colours.json"

PEN_LENGTH = 0.13  # metres
MIN_ASPECT = 3.0  # blob must be at least this many times longer than wide
MIN_PIXELS = 150  # at 1280x960
MAX_COLOUR_DIST = 3.5  # Mahalanobis distance (in colour-model standard deviations)
EDGE_PX = 4  # a pen tip this close to the frame edge counts as cut off
MIN_LEN, MAX_LEN = 0.05, 0.20  # metres: plausible measured lengths for a 13 cm pen


# Classes that are NOT pens: everything else a pixel can be matched to instead.
# "other" = white/grey/black things (arm, radiator, cables, wall). Without it, a pen whose
# colour is nearly neutral (Ben's rose-gold pen: Lab a/b ~ (135, 128), white ~ (128, 128))
# matches white plastic better than wood, and the arm shows up as "pink pen".
NON_PEN = {"table", "other"}


def chroma(img_rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB -> (H, W, 2) Lab a/b: colour without brightness."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)[..., 1:].astype(np.float32)


def pixel_features(img_rgb: np.ndarray, kind: str) -> np.ndarray:
    """"ab" = chroma only (shadow-proof; enough in sim). "lab" = chroma + brightness, needed
    on the real rig, where brightness is what separates the pale pen from white plastic."""
    return chroma(img_rgb) if kind == "ab" else cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab).astype(np.float32)


class ColourModel:
    """A Gaussian over pixel colour per class: 'this is what blue-pen pixels look like'."""

    def __init__(self, classes: dict[str, dict], features: str = "ab"):
        self.classes = classes  # name -> {"mean": [...], "icov": dxd}
        self.features = features

    @property
    def pens(self) -> list[str]:
        return [c for c in self.classes if c not in NON_PEN]

    @classmethod
    def fit(cls, samples: dict[str, np.ndarray], features: str = "ab") -> ColourModel:
        out = {}
        for name, x in samples.items():
            cov = np.cov(x.T) + np.eye(x.shape[1]) * 0.5  # floor, so near-constant colours stay usable
            out[name] = {"mean": x.mean(0).tolist(), "icov": np.linalg.inv(cov).tolist(), "n": len(x)}
        return cls(out, features)

    def distance(self, x: np.ndarray, name: str) -> np.ndarray:
        c = self.classes[name]
        d = x - np.array(c["mean"], np.float32)
        return np.sqrt(np.einsum("...i,ij,...j->...", d, np.array(c["icov"], np.float32), d))

    def save(self, path: pathlib.Path = MODEL_FILE):
        path.write_text(json.dumps({"features": self.features, "classes": self.classes}, indent=1))

    @classmethod
    def load(cls, path: pathlib.Path = MODEL_FILE) -> ColourModel:
        d = json.loads(path.read_text())
        if "classes" not in d:  # older files: a bare chroma-only class dict
            return cls(d, "ab")
        return cls(d["classes"], d["features"])


def pixel_to_table(calib, uv: np.ndarray, w: int, h: int, z: float) -> np.ndarray:
    """Pixels (N, 2) -> table points (N, 3) at height z, via whichever calibration is in use."""
    return calib.to_table(uv, w, h, z)


def detect_pens(img_rgb: np.ndarray, calib, model: ColourModel, pen_z: float,
                workspace: tuple[tuple[float, float], tuple[float, float]]) -> list[dict]:
    """Every pen-shaped blob of a known pen colour, as table coordinates."""
    h, w = img_rgb.shape[:2]
    x = pixel_features(cv2.GaussianBlur(img_rgb, (5, 5), 0), model.features)
    dists = {c: model.distance(x, c) for c in model.classes}
    found = []
    for colour in model.pens:
        others = np.min([dists[c] for c in model.classes if c != colour], axis=0)
        mask = ((dists[colour] < MAX_COLOUR_DIST) & (dists[colour] < others)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < MIN_PIXELS * (w * h) / (1280 * 960):
                continue
            ys, xs = np.nonzero(labels == i)
            pts = np.column_stack([xs, ys]).astype(np.float64)
            # Principal axis of the blob; its two extreme points are the pen's tips.
            mean = pts.mean(0)
            _, s, vt = np.linalg.svd(pts - mean, full_matrices=False)
            if s[0] / max(s[1], 1e-6) < MIN_ASPECT:
                continue
            t = (pts - mean) @ vt[0]
            ends_px = np.array([mean + t.min() * vt[0], mean + t.max() * vt[0]])
            ends = pixel_to_table(calib, ends_px, w, h, pen_z)
            # A tip within a few pixels of the frame edge is probably cut off by the edge.
            # Then only the other tip is trustworthy: measure half a pen from it instead.
            cut = [bool(u < EDGE_PX or v < EDGE_PX or u > w - 1 - EDGE_PX or v > h - 1 - EDGE_PX)
                   for u, v in ends_px]
            centre = ends.mean(0)
            if cut[0] != cut[1]:
                good, bad = (ends[1], ends[0]) if cut[0] else (ends[0], ends[1])
                axis = (bad - good) / max(np.linalg.norm(bad - good), 1e-9)
                centre = good + axis * PEN_LENGTH / 2
            (x0, x1), (y0, y1) = workspace
            if not (x0 <= centre[0] <= x1 and y0 <= centre[1] <= y1):
                continue
            length = float(np.linalg.norm(ends[1, :2] - ends[0, :2]))
            # A 13 cm pen measures 13 cm. Short blobs are cable glints, shadows and wood
            # grain (the real rig produced dozens at 1-6 cm). A pen cut off by the frame
            # edge is allowed to look short; one hidden under the arm is not (go_home first).
            if not (MIN_LEN <= length <= MAX_LEN or (cut[0] != cut[1] and length <= MAX_LEN)):
                continue
            yaw = float(np.degrees(np.arctan2(ends[1, 1] - ends[0, 1], ends[1, 0] - ends[0, 0])))
            yaw = (yaw + 90) % 180 - 90  # a pen pointing either way is the same pen
            found.append({
                "colour": colour, "x": round(float(centre[0]), 4), "y": round(float(centre[1]), 4),
                "yaw_deg": round(yaw, 1), "length_m": round(length, 3),
                # Partly hidden (under the arm, or off the frame edge) = centre less reliable.
                "fully_visible": bool(abs(length - PEN_LENGTH) < 0.01 and not any(cut)),
                "centre_from_one_end": bool(cut[0] != cut[1]),
                "_px": ends_px, "_pixels": int(len(pts)),
            })
    found.sort(key=lambda d: (d["colour"], d["y"]))
    for i, d in enumerate(found, 1):
        d["id"] = i
    return found


def annotate(img_rgb: np.ndarray, found: list[dict]) -> np.ndarray:
    """Draw each detection: its axis, both tips, and a numbered label."""
    out = img_rgb.copy()
    for d in found:
        (u0, v0), (u1, v1) = d["_px"].astype(int)
        cv2.line(out, (u0, v0), (u1, v1), (0, 255, 0), 2, cv2.LINE_AA)
        for u, v in ((u0, v0), (u1, v1)):
            cv2.circle(out, (u, v), 6, (0, 255, 0), 2, cv2.LINE_AA)
        cu, cv_ = (u0 + u1) // 2, (v0 + v1) // 2
        cv2.drawMarker(out, (cu, cv_), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
        label = f"#{d['id']} {d['colour']} ({d['x']:.3f}, {d['y']:+.3f}) {d['yaw_deg']:+.0f}deg"
        if d["centre_from_one_end"]:
            label += " (off-frame: centre from far tip)"
        elif not d["fully_visible"]:
            label += " PARTLY HIDDEN"
        for col, th in (((0, 0, 0), 5), ((255, 255, 255), 2)):
            cv2.putText(out, label, (cu + 12, cv_ - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, th, cv2.LINE_AA)
    return out


def public(found: list[dict]) -> list[dict]:
    """Detections without the private drawing fields, for the tool result."""
    return [{k: v for k, v in d.items() if not k.startswith("_")} for d in found]
