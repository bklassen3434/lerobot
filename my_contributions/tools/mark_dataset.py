"""Turn a colour-prompted dataset into a marker-prompted one (see marker_overlay.py).

For each episode: detect the named pen (task "blue"/"pink") in the first top-camera frame,
draw a green ring there on EVERY top frame, and replace the task with MARKED_TASK. Actions,
state and the wrist camera are copied unchanged. No re-recording needed.

    # check only: detect every episode, write a contact sheet, write nothing else
    uv run python my_contributions/tools/mark_dataset.py bklassen3434/pick_pen_v2_trimmed

    # write the marked copy
    uv run python my_contributions/tools/mark_dataset.py bklassen3434/pick_pen_v2_trimmed \
        --out bklassen3434/pick_pen_v2_marked

Sanity check: BOTH pens must be found in every episode, clearly apart. Any episode that
fails is reported and (when writing) skipped, never guessed. The other pen's position is
saved too (meta/markers.json), so the offline probe can move the ring onto it. Look at the contact sheet
before training: each ring must sit on the pen the task names.
"""

import argparse
import json
import pathlib

import cv2
import numpy as np
import torch
from marker_overlay import MARKED_TASK, draw_marker, find_pen_px

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CACHE = pathlib.Path("/Users/benklassen/.cache/huggingface/lerobot")
TOP = "observation.images.top"
OTHER = {"blue": "pink", "pink": "blue"}
MIN_SEPARATION_PX = 80  # the two pens are ~170 px apart in the top camera


def to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    """Dataset images arrive as CHW float in [0,1]; drawing and writing want HWC uint8."""
    return (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def detect_all(src: LeRobotDataset) -> list[dict]:
    """Find BOTH pens in each episode's first frame. Both must be found, clearly apart."""
    rows = []
    for ep in range(src.meta.total_episodes):
        item = src[int(src.meta.episodes["dataset_from_index"][ep])]
        img = to_hwc_uint8(item[TOP])
        other = OTHER[item["task"]]
        uv, other_uv = find_pen_px(img, item["task"]), find_pen_px(img, other)
        r = {"ep": ep, "task": item["task"], "uv": uv, "other_uv": other_uv, "img": img, "ok": False}
        if uv is None:
            r["why"] = f"no {item['task']} pen found"
        elif other_uv is None:
            r["why"] = f"no {other} pen found"
        elif np.hypot(uv[0] - other_uv[0], uv[1] - other_uv[1]) < MIN_SEPARATION_PX:
            r["why"] = "both pens found in the same place"
        else:
            r["ok"], r["why"] = True, ""
        rows.append(r)
    bad = [r for r in rows if not r["ok"]]
    print(f"{len(rows) - len(bad)}/{len(rows)} episodes OK" + ("" if not bad else ", problems:"))
    for r in bad:
        print(f"  ep {r['ep']:3d} ({r['task']}): {r['why']}")
    return rows


def contact_sheet(rows: list[dict], path: pathlib.Path, cols: int = 6) -> None:
    """Every episode's first frame with its ring, labelled; problems get a red border."""
    tiles = []
    for r in rows:
        img = draw_marker(r["img"], r["uv"]) if r["uv"] else r["img"].copy()
        img = cv2.resize(img, (320, 240))
        label = f"ep{r['ep']} {r['task']}" + ("" if r["ok"] else f" !! {r['why']}")
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(img, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th, cv2.LINE_AA)
        if not r["ok"]:
            cv2.rectangle(img, (0, 0), (319, 239), (255, 0, 0), 6)
        tiles.append(img)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"contact sheet: {path}")


def write(src: LeRobotDataset, rows: list[dict], out_repo: str) -> None:
    features = {k: v for k, v in src.meta.features.items()
                if not k.startswith(("index", "timestamp", "frame_index", "episode_index", "task_index"))}
    dst = LeRobotDataset.create(out_repo, int(src.meta.fps), features=features, root=CACHE / out_repo,
                                robot_type=src.meta.robot_type, use_videos=True)
    markers = {}
    for r in rows:
        if not r["ok"]:
            print(f"  ep {r['ep']:3d}: SKIPPED ({r['why']})")
            continue
        lo = int(src.meta.episodes["dataset_from_index"][r["ep"]])
        hi = int(src.meta.episodes["dataset_to_index"][r["ep"]])
        for i in range(lo, hi):
            item = src[i]
            frame = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in item.items() if k in features}
            for k in features:
                if "image" in k:
                    frame[k] = to_hwc_uint8(item[k])
            frame[TOP] = draw_marker(frame[TOP], r["uv"])
            frame["task"] = MARKED_TASK
            dst.add_frame(frame)
        dst.save_episode()
        markers[dst.meta.total_episodes - 1] = {"source_ep": r["ep"], "colour": r["task"],
                                                "uv": [round(c, 1) for c in r["uv"]],
                                                "other_uv": [round(c, 1) for c in r["other_uv"]]}
        print(f"  ep {r['ep']:3d} -> {dst.meta.total_episodes - 1:3d} ({r['task']}, {hi - lo} frames)", flush=True)
    dst.finalize()
    # Where each ring is, and where the other pen is: the offline probe swaps between them.
    (CACHE / out_repo / "meta" / "markers.json").write_text(json.dumps(markers, indent=1))
    print(f"\nwrote {out_repo}: {dst.meta.total_episodes} episodes, {dst.meta.total_frames} frames")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("--out", default=None, help="write the marked copy under this repo id")
    p.add_argument("--sheet", default=None, help="contact sheet path (default: <repo>_markers.jpg in cwd)")
    a = p.parse_args()

    src = LeRobotDataset(a.repo_id, root=CACHE / a.repo_id)
    rows = detect_all(src)
    contact_sheet(rows, pathlib.Path(a.sheet or f"{a.repo_id.split('/')[-1]}_markers.jpg"))
    if a.out:
        write(src, rows, a.out)


if __name__ == "__main__":
    main()
