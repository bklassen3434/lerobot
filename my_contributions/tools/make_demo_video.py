"""Turn marker-model rollouts into a demo video: instruction -> Claude -> ring -> SmolVLA pick.

Each attempt gets a short caption card (what was typed, what Claude chose), then the top camera
(with the ring the policy actually saw) and the wrist camera side by side, in real time.

Frames are re-timed to wall-clock: sentry stamps frames at 30 fps, but the loop really ran at
~22 Hz (15 s of attempt = ~330 frames), so playing the frames at 30 fps would look sped up.

Usage (from the hyderabad-v1 workspace):
    # instructions and outcomes come from the log marker_rollout.sh keeps next to the dataset
    uv run --no-sync python my_contributions/tools/make_demo_video.py bklassen3434/rollout_marker_<stamp>
    # only some attempts (1-based); the end card then says how many of how many are shown
    uv run --no-sync python my_contributions/tools/make_demo_video.py bklassen3434/rollout_marker_<stamp> --episodes 1 2 4
    # older datasets without a log: pass every instruction in order (no score on the end card)
    uv run --no-sync python my_contributions/tools/make_demo_video.py <repo> "pick up the pink pen" "..."
"""

import argparse
import pathlib
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CACHE = pathlib.Path("/Users/benklassen/.cache/huggingface/lerobot")
ATTEMPT_S = 15.0  # wall-clock length of every attempt (marker_rollout.sh DURATION)
FPS = 30
W, H = 1280, 720
CAM_Y = 130  # cameras sit between the header and the pipeline strip
FONT = "/System/Library/Fonts/Helvetica.ttc"
PEN_RGB = {"pink": (232, 150, 140), "blue": (110, 160, 230)}
BG, FG, DIM, GREEN = (18, 18, 22), (240, 240, 240), (140, 140, 150), (0, 220, 90)
STEPS = ["You type an instruction", "Claude picks the colour", "Detector rings that pen", "SmolVLA picks it up"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def pipeline_strip(d: ImageDraw.ImageDraw, active: int) -> None:
    """The four stages along the bottom; stages up to `active` are lit."""
    y, bw = H - 70, W // 4
    for i, label in enumerate(STEPS):
        x = i * bw
        lit = i <= active
        d.rounded_rectangle((x + 10, y, x + bw - 10, y + 50), 10,
                            fill=(30, 70, 45) if lit else (35, 35, 42),
                            outline=GREEN if i == active else None, width=2)
        d.text((x + bw // 2, y + 25), f"{i + 1}. {label}", font=font(17, lit), fill=FG if lit else DIM, anchor="mm")


def header(d: ImageDraw.ImageDraw, n: int, total: int, instruction: str, colour: str, show_claude: bool) -> None:
    d.text((30, 22), f"Attempt {n}/{total}", font=font(20), fill=DIM)
    d.text((30, 52), "You:", font=font(30, True), fill=DIM)
    d.text((105, 52), f"“{instruction}”", font=font(30), fill=FG)
    if show_claude:
        d.text((W - 330, 52), "Claude:", font=font(30, True), fill=DIM)
        d.rounded_rectangle((W - 175, 48, W - 35, 92), 22, fill=PEN_RGB[colour])
        d.text((W - 105, 70), colour, font=font(28, True), fill=(20, 20, 20), anchor="mm")


def cams(img: Image.Image, top: np.ndarray, wrist: np.ndarray, label: bool = True) -> None:
    for x, frame, name in ((0, top, "top camera  (green ring = what SmolVLA is told to grab)"),
                           (640, wrist, "wrist camera")):
        img.paste(Image.fromarray(cv2.resize(frame, (640, 480))), (x, CAM_Y))
        if label:
            ImageDraw.Draw(img).text((x + 12, CAM_Y + 10), name, font=font(16, True), fill=FG,
                                     stroke_width=3, stroke_fill=(0, 0, 0))


def title_card(lines: list[tuple[str, int, tuple]], seconds: float) -> list[np.ndarray]:
    img, d = canvas()
    lines = [line for line in lines if line[0]]
    y = H // 2 - sum(s + 18 for _, s, _ in lines) // 2
    for text, size, col in lines:
        d.text((W // 2, y), text, font=font(size, size >= 40), fill=col, anchor="mt")
        y += size + 18
    return [np.asarray(img)] * int(seconds * FPS)


def to_hwc(t) -> np.ndarray:
    return (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def attempt_frames(ds: LeRobotDataset, ep: int, n: int, total: int, instruction: str, colour: str) -> list[np.ndarray]:
    lo = int(ds.meta.episodes["dataset_from_index"][ep])
    hi = int(ds.meta.episodes["dataset_to_index"][ep])
    count = hi - lo
    state = np.stack([np.asarray(ds.hf_dataset[i]["observation.state"]) for i in range(lo, hi)])
    # Stop one second after the arm has parked, rather than showing it sitting still.
    moving = np.where(np.abs(np.diff(state, axis=0)).sum(1) > 0.5)[0]
    end_s = min(ATTEMPT_S, (moving[-1] + 1) / count * ATTEMPT_S + 1.0) if len(moving) else ATTEMPT_S

    cache: dict[int, tuple] = {}

    def frame_at(t: float):
        i = min(int(t / ATTEMPT_S * count), count - 1)
        if i not in cache:
            item = ds[lo + i]
            cache.clear()
            cache[i] = (to_hwc(item["observation.images.top"]), to_hwc(item["observation.images.wrist"]))
        return cache[i]

    out = []
    first_top, first_wrist = frame_at(0)
    # Stage cards over the first frame: typed -> Claude -> ring.
    for stage, secs in ((0, 1.2), (1, 1.2), (2, 1.4)):
        img, d = canvas()
        header(d, n, total, instruction, colour, show_claude=stage >= 1)
        cams(img, first_top if stage >= 2 else _without_ring_hint(first_top), first_wrist)
        pipeline_strip(d, stage)
        out += [np.asarray(img)] * int(secs * FPS)
    # The pick itself, in real time.
    for k in range(int(end_s * FPS)):
        t = k / FPS
        top, wrist = frame_at(t)
        img, d = canvas()
        header(d, n, total, instruction, colour, show_claude=True)
        cams(img, top, wrist)
        pipeline_strip(d, 3)
        d.text((W - 30, CAM_Y + 492), f"{t:4.1f} s  (real time)", font=font(16), fill=DIM, anchor="rt")
        out.append(np.asarray(img))
    return out


def _without_ring_hint(top: np.ndarray) -> np.ndarray:
    """Before the 'ring' stage, dim the frame so the ring's arrival reads as a step."""
    return (top * 0.55).astype(np.uint8)


def read_log(root: pathlib.Path) -> dict[int, dict]:
    """meta/instructions.tsv written by marker_rollout.sh: episode, instruction, colour, right, lifted."""
    f = root / "meta" / "instructions.tsv"
    if not f.exists():
        return {}
    rows = {}
    for line in f.read_text().splitlines():
        ep, instruction, colour, right, lifted = line.split("\t")
        rows[int(ep)] = {"instruction": instruction, "colour": colour,
                         "right": right.lower().startswith("y"), "lifted": lifted.lower().startswith("y")}
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("instructions", nargs="*",
                   help="what was typed for each attempt (default: meta/instructions.tsv from marker_rollout.sh)")
    p.add_argument("--episodes", type=int, nargs="+", help="which attempts to include, 1-based (default: all)")
    p.add_argument("--out", default="my_contributions/media/demo_marker.mp4")
    a = p.parse_args()

    ds = LeRobotDataset(a.repo_id, root=CACHE / a.repo_id)
    total = ds.meta.total_episodes
    log = read_log(ds.root)
    if a.instructions:
        if len(a.instructions) != total:
            sys.exit(f"{total} attempts in {a.repo_id}, but {len(a.instructions)} instructions given")
        log = {ep: {"instruction": s, "colour": next(c for c in PEN_RGB if c in s.lower()), "right": None,
                    "lifted": None} for ep, s in enumerate(a.instructions)}
    missing = [ep + 1 for ep in range(total) if ep not in log]
    if missing:
        sys.exit(f"no instruction logged for attempt(s) {missing}: pass them all on the command line")

    eps = [e - 1 for e in a.episodes] if a.episodes else list(range(total))
    everything = len(eps) == total
    frames = title_card([("Pick the pen I ask for", 52, FG),
                         ("Claude chooses  ·  a green ring marks it  ·  SmolVLA picks it up", 26, DIM),
                         ("SO-101 arm  ·  real robot, real time" + (", every attempt" if everything else ""), 20, DIM)],
                        3.0)
    for n, ep in enumerate(eps, 1):
        frames += attempt_frames(ds, ep, n, len(eps), log[ep]["instruction"], log[ep]["colour"])
        print(f"  attempt {ep + 1} rendered ({n}/{len(eps)})", flush=True)

    # End card from what was actually observed, never assumed.
    if all(log[ep]["right"] is not None for ep in eps):
        right = sum(log[ep]["right"] for ep in eps)
        lifted = sum(log[ep]["lifted"] for ep in eps)
        result = f"Right pen {right}/{len(eps)}  ·  picked up {lifted}/{len(eps)}"
        note = "" if everything else f"({len(eps)} of {total} recorded attempts shown)"
    else:
        result, note = "", "outcomes not logged"
        print("  outcomes not logged, so the end card shows no score")
    frames += title_card([(result, 48, GREEN), (note, 20, DIM),
                          ("SmolVLA never sees a word of the instruction, only the ring.", 24, DIM)], 3.5)

    # Write raw frames to ffmpeg: H.264 + yuv420p plays everywhere (QuickTime, Slack, phones).
    out = pathlib.Path(a.out)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "20", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"wrote {out}: {len(frames) / FPS:.1f} s")


if __name__ == "__main__":
    main()
