"""Decide which pen to pick and where the ring goes. Run before each marker-model attempt.

    1. Claude turns a free-form instruction ("grab the pink one") into a colour: blue or pink.
       (--colour skips this.)
    2. One frame is grabbed from the top camera (arm parked, both pens in view).
    3. The detector finds BOTH pens; the ring goes on the chosen one. If either pen is missing
       it stops rather than guessing.
    4. A preview with the ring is saved, and the ring position and colour are printed as
       MARKER_UV=u,v and COLOUR=c for marker_rollout.sh.

Usage (from the hyderabad-v1 workspace):
    uv run --no-sync python my_contributions/tools/marker_aim.py "pick up the pink pen"
    uv run --no-sync python my_contributions/tools/marker_aim.py --colour blue
    uv run --no-sync python my_contributions/tools/marker_aim.py "the blue one" --image frame.png   # no camera
"""

import argparse
import pathlib
import subprocess
import sys
import time

import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from marker_overlay import draw_marker, find_pen_px  # noqa: E402

COLOURS = ("blue", "pink")
TOP_CAMERA = 0  # see real_config.json; camera 2 is the MacBook webcam
WIDTH, HEIGHT = 640, 480
WAKE_TIMEOUT_S = 8.0  # how long to wait for the camera to start sending real images
SETTLE_FRAMES = 30  # then ~1 s of real frames so auto-exposure settles
PREVIEW = pathlib.Path(__file__).resolve().parents[1] / "agent" / "out" / "marker_aim.jpg"

PROMPT = """A robot arm has two pens in front of it: a blue pen and a pink (rose-gold) pen.
The user said: "{instruction}"
Which pen should the robot pick up? Answer with exactly one word: blue, pink, or none
(none if the instruction doesn't clearly pick one)."""


def ask_claude(instruction: str) -> str:
    # stdin=DEVNULL: `claude -p` from a subprocess otherwise waits on stdin forever.
    out = subprocess.run(["claude", "-p", "--model", "haiku", PROMPT.format(instruction=instruction)],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    if out.returncode != 0:
        raise SystemExit(f"claude failed: {out.stderr.strip()}")
    word = out.stdout.strip().lower().strip(".!\"' ")
    if word not in COLOURS:
        raise SystemExit(f"Claude couldn't pick a pen from {instruction!r} (said {out.stdout.strip()!r})")
    return word


def grab_top_frame():
    """One settled frame. USB cameras on macOS send black frames for a second or two after
    opening, then auto-exposure ramps up, so wait for real images, then let exposure settle."""
    cap = cv2.VideoCapture(TOP_CAMERA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    try:
        deadline, lit, bgr = time.time() + WAKE_TIMEOUT_S, 0, None
        while time.time() < deadline and lit < SETTLE_FRAMES:
            ok, frame = cap.read()
            if ok and frame is not None:
                bgr = frame
                lit = lit + 1 if frame.max() >= 20 else 0
    finally:
        cap.release()
    if bgr is None:
        raise SystemExit(f"could not read top camera (index {TOP_CAMERA})")
    if bgr.shape[:2] != (HEIGHT, WIDTH):
        raise SystemExit(f"top camera gave {bgr.shape[1]}x{bgr.shape[0]}, expected {WIDTH}x{HEIGHT}")
    if lit < SETTLE_FRAMES:
        raise SystemExit(f"the top camera was still black after {WAKE_TIMEOUT_S:.0f} s. Check the camera "
                         "index, lens cap, and that this terminal has camera permission.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # lerobot cameras (and the dataset) are RGB


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("instruction", nargs="?", default=None)
    p.add_argument("--colour", choices=COLOURS, help="skip Claude and use this colour")
    p.add_argument("--image", help="use this image instead of the camera (for testing)")
    a = p.parse_args()
    if not (a.colour or a.instruction):
        p.error("give an instruction or --colour")

    colour = a.colour or ask_claude(a.instruction)
    print(f"colour: {colour}" + ("" if a.colour else f"   (Claude, from {a.instruction!r})"), file=sys.stderr)

    if a.image:
        img = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
    else:
        img = grab_top_frame()
    other = next(c for c in COLOURS if c != colour)
    uv, other_uv = find_pen_px(img, colour), find_pen_px(img, other)
    if uv is None or other_uv is None:
        missing = [c for c, x in ((colour, uv), (other, other_uv)) if x is None]
        cv2.imwrite(str(PREVIEW), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        raise SystemExit(f"can't see the {' and '.join(missing)} pen (frame saved to {PREVIEW}). "
                         "Is the arm parked out of the way and are both pens on the table?")

    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PREVIEW), cv2.cvtColor(draw_marker(img, uv), cv2.COLOR_RGB2BGR))
    print(f"ring on the {colour} pen at ({uv[0]:.0f}, {uv[1]:.0f}); preview: {PREVIEW}", file=sys.stderr)
    # stdout is read by marker_rollout.sh: exactly these two lines.
    print(f"MARKER_UV={uv[0]:.1f},{uv[1]:.1f}")
    print(f"COLOUR={colour}")


if __name__ == "__main__":
    main()
