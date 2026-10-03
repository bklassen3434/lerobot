"""Put the top camera back exactly where it was when the training data was recorded.

The camera did not move by more than ~1 px across all 60 training episodes, so the marker
model has only ever seen ONE view of the table. This shows the live top camera against a
reference frame from the dataset and measures, live, how far off it is:

    shift  : how many pixels the scene sits away from where it should be
    rotate : roll of the image, degrees
    zoom   : scene bigger (>100%) or smaller than it should be (camera too close / too far)

It also hints which way to nudge the camera, and shows whether the pen detector sees both pens.

How the numbers are made: ORB features (corners on the radiator, cables, table grain) are
matched between the reference and the live frame, and a shift+rotation+zoom transform is
fitted with RANSAC, so a few things moving (a cable, the pens) don't throw it off.

Usage (from the hyderabad-v1 workspace; park the arm first, like in the reference):
    uv run --no-sync python my_contributions/tools/camera_align.py
Keys:  v = cycle view (edges / blend / live / reference)   s = save snapshot   q = quit
Test without the camera:  --image some.png [--once]
"""

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from marker_overlay import draw_marker, find_pen_px  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "agent" / "out"
REF_FILE = OUT / "camera_reference.png"
REF_REPO, REF_EP = "bklassen3434/pick_pen_v2_trimmed", 59  # arm parked, both pens in view
CACHE = "/Users/benklassen/.cache/huggingface/lerobot"
TOP_CAMERA, W, H = 0, 640, 480

# Tolerances. The training data was within ~1 px, so we don't know how much drift the policy
# tolerates. "good" is a cautious guess, not a measured limit.
GOOD = {"shift": 4.0, "rot": 0.4, "zoom": 0.01}
CLOSE = {"shift": 12.0, "rot": 1.5, "zoom": 0.03}
VIEWS = ("edges", "blend", "live", "reference")


def load_reference() -> np.ndarray:
    """First top frame of a training episode, cached as a PNG (RGB in memory)."""
    if not REF_FILE.exists():
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(REF_REPO, root=f"{CACHE}/{REF_REPO}")
        t = ds[int(ds.meta.episodes["dataset_from_index"][REF_EP])]["observation.images.top"]
        OUT.mkdir(parents=True, exist_ok=True)
        rgb = (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        cv2.imwrite(str(REF_FILE), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(cv2.imread(str(REF_FILE)), cv2.COLOR_BGR2RGB)


class Aligner:
    def __init__(self, ref: np.ndarray):
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kr, self.dr = self.orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), None)

    def measure(self, live: np.ndarray) -> dict | None:
        """Transform taking reference pixels to live pixels, as shift / rotation / zoom."""
        kl, dl = self.orb.detectAndCompute(cv2.cvtColor(live, cv2.COLOR_RGB2GRAY), None)
        if dl is None or len(kl) < 20:
            return None
        m = self.bf.match(self.dr, dl)
        if len(m) < 20:
            return None
        pr = np.float32([self.kr[x.queryIdx].pt for x in m])
        pl = np.float32([kl[x.trainIdx].pt for x in m])
        M, inl = cv2.estimateAffinePartial2D(pr, pl, method=cv2.RANSAC, ransacReprojThreshold=3)
        if M is None or inl.sum() < 15:
            return None
        # Measure the shift at the image centre, so rotation/zoom don't leak into it.
        c = M @ np.array([W / 2, H / 2, 1.0])
        return {"dx": c[0] - W / 2, "dy": c[1] - H / 2,
                "rot": float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
                "zoom": float(np.hypot(M[0, 0], M[1, 0])), "inliers": int(inl.sum()), "M": M}


def grade(r: dict) -> str:
    shift, rot, zoom = np.hypot(r["dx"], r["dy"]), abs(r["rot"]), abs(r["zoom"] - 1)
    if shift <= GOOD["shift"] and rot <= GOOD["rot"] and zoom <= GOOD["zoom"]:
        return "GOOD"
    if shift <= CLOSE["shift"] and rot <= CLOSE["rot"] and zoom <= CLOSE["zoom"]:
        return "CLOSE"
    return "OFF"


def hints(r: dict) -> list[str]:
    """Which way to nudge the camera. Turning a camera moves the scene the OPPOSITE way in the
    image: pan right and the scene slides left. So a scene sitting too far right is fixed by
    panning right, a scene too low by tilting down, and an image rotated clockwise by rolling
    the camera clockwise."""
    out = []
    if abs(r["dx"]) > GOOD["shift"] / 2:
        out.append(f"pan camera {'RIGHT' if r['dx'] > 0 else 'LEFT'} (scene is {abs(r['dx']):.0f}px too far "
                   f"{'right' if r['dx'] > 0 else 'left'})")
    if abs(r["dy"]) > GOOD["shift"] / 2:
        out.append(f"tilt camera {'DOWN' if r['dy'] > 0 else 'UP'} (scene is {abs(r['dy']):.0f}px too "
                   f"{'low' if r['dy'] > 0 else 'high'})")
    if abs(r["rot"]) > GOOD["rot"] / 2:
        out.append(f"roll camera {'CLOCKWISE' if r['rot'] > 0 else 'ANTI-CLOCKWISE'} "
                   f"(image is rotated {abs(r['rot']):.1f} deg {'clockwise' if r['rot'] > 0 else 'anti-clockwise'})")
    if abs(r["zoom"] - 1) > GOOD["zoom"] / 2:
        out.append(f"move camera {'AWAY from' if r['zoom'] > 1 else 'CLOSER to'} the table "
                   f"(scene is {abs(r['zoom'] - 1) * 100:.0f}% too {'big' if r['zoom'] > 1 else 'small'})")
    return out


def render(ref, live, ref_edges, r, pens, view) -> np.ndarray:
    if view == "edges":
        img = live.copy()
        img[ref_edges > 0] = (0, 255, 0)  # where the reference's lines are
    elif view == "blend":
        img = cv2.addWeighted(live, 0.5, ref, 0.5, 0)
    else:
        img = (live if view == "live" else ref).copy()
    for colour, uv in pens.items():
        if uv is not None:
            img = draw_marker(img, uv)
    panel = np.full((H, 420, 3), 30, np.uint8)
    lines = [(f"view: {view}  (v = next view, s = save, q = quit)", (200, 200, 200))]
    if r is None:
        lines.append(("can't match the reference - way off, or too dark?", (255, 80, 80)))
    else:
        g = grade(r)
        col = {"GOOD": (80, 255, 80), "CLOSE": (255, 200, 0), "OFF": (255, 80, 80)}[g]
        lines += [(g, col),
                  (f"shift  {np.hypot(r['dx'], r['dy']):5.1f} px  (dx {r['dx']:+.1f}, dy {r['dy']:+.1f})", col),
                  (f"rotate {r['rot']:+5.2f} deg", col),
                  (f"zoom   {r['zoom'] * 100:6.1f} %", col),
                  (f"matches {r['inliers']}", (150, 150, 150)), ("", col)]
        lines += [(h, (255, 255, 255)) for h in hints(r)] or [("hold it there!", (80, 255, 80))]
    lines.append(("", (0, 0, 0)))
    for colour, uv in pens.items():
        lines.append((f"{colour} pen: {'found' if uv else 'NOT FOUND'}", (80, 255, 80) if uv else (255, 80, 80)))
    for i, (text, col) in enumerate(lines):
        cv2.putText(panel, text, (12, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return np.hstack([img, panel])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="use this still image instead of the camera")
    p.add_argument("--once", action="store_true", help="print one measurement and exit (no window)")
    a = p.parse_args()

    ref = load_reference()
    ref_edges = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), 60, 160)
    aligner = Aligner(ref)

    cap = None
    if not a.image:
        cap = cv2.VideoCapture(TOP_CAMERA)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    view, last_measure, r, pens = 0, 0.0, None, {}
    started = time.time()  # the first frames can be dark while exposure settles
    while True:
        if cap is not None:
            ok, bgr = cap.read()
            if not ok:
                raise SystemExit(f"could not read top camera (index {TOP_CAMERA}). If OpenCV said 'not authorized', "
                                 "this terminal app has no camera permission: run it from Terminal/iTerm instead, "
                                 "or allow it under System Settings > Privacy & Security > Camera.")
        else:
            bgr = cv2.imread(a.image)
        if bgr.shape[:2] != (H, W):
            raise SystemExit(f"frame is {bgr.shape[1]}x{bgr.shape[0]}, expected {W}x{H}")
        if bgr.max() < 20 and cap is not None and time.time() - started > 3:
            raise SystemExit("the top camera returned a black frame. Usually this terminal has no camera permission: "
                         "run from the macOS Terminal app, or allow this app under System Settings > "
                         "Privacy & Security > Camera. Otherwise check the lens cap / camera index.")
        live = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if time.time() - last_measure > 0.2 or a.once:  # ~5 measurements/s keeps the feed smooth
            r = aligner.measure(live)
            pens = {c: find_pen_px(live, c) for c in ("blue", "pink")}
            last_measure = time.time()

        if a.once:
            if r is None:
                print("no match")
            else:
                print(f"{grade(r)}: shift ({r['dx']:+.1f}, {r['dy']:+.1f}) px, rotate {r['rot']:+.2f} deg, "
                      f"zoom {r['zoom'] * 100:.1f}%, {r['inliers']} matches")
                print("\n".join(hints(r)) or "hold it there!")
            print("pens:", {c: ("found" if uv else "NOT FOUND") for c, uv in pens.items()})
            return

        frame = render(ref, live, ref_edges, r, pens, VIEWS[view])
        cv2.imshow("camera alignment", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("v"):
            view = (view + 1) % len(VIEWS)
        if key == ord("s"):
            path = OUT / f"camera_align_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print("saved", path)
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
