"""Step 1a of the V-JEPA 2 world-model experiment: turn the pen dataset into V-JEPA 2 embeddings.

For every STRIDE-th frame of a camera (top by default; `--cam wrist` writes embeddings_wrist.npz), feed the 2-frame clip (previous frame, this frame)
through the frozen V-JEPA 2 ViT-L encoder and save:
  - `mean`:   the 16x16 patch tokens averaged into one 1024-d vector ("what is in the scene")
  - `grid`:   the tokens average-pooled to a 4x4 grid x 1024 ("what is WHERE in the scene")
  - `pixels`: the frame shrunk to 32x24 RGB, a dumb baseline the embeddings have to beat
plus the labels probe.py needs (joint state, task, which side each pen is on).

    cd /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1 && \
        .venv/bin/python my_contributions/vjepa/embed.py

Frames are squashed (not centre-cropped) to 256x256: the stock V-JEPA 2 processor centre-crops,
which would cut the outer ~20% off each side of the 640x480 frame, where the pens can sit.
"""

import argparse
import glob
import pathlib
import sys
import time

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import VJEPA2Model

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from marker_overlay import find_pen_px  # noqa: E402

CACHE = pathlib.Path("/Users/benklassen/.cache/huggingface/lerobot")
OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
FPS = 30
RES = 256
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def load_meta(root: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/*/*.parquet"))))
    data = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/*/*.parquet"))))
    return eps.reset_index(drop=True), data.set_index(["episode_index", "frame_index"]).sort_index()


def iter_frames(root: pathlib.Path, eps: pd.DataFrame, cam: str = "top"):
    """Yield (episode, frame_index, rgb uint8 HxWx3) for every frame, decoding each video file once."""
    cam = f"observation.images.{cam}"
    for (chunk, file), group in eps.groupby([f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"]):
        path = root / f"videos/{cam}/chunk-{chunk:03d}/file-{file:03d}.mp4"
        spans = group[["episode_index", f"videos/{cam}/from_timestamp", "length"]].to_numpy()
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                t = float(frame.pts * stream.time_base)
                for ep, start, length in spans:
                    idx = round((t - start) * FPS)
                    if 0 <= idx < length:
                        yield int(ep), idx, frame.to_ndarray(format="rgb24")
                        break


def to_clip(prev: np.ndarray, cur: np.ndarray) -> torch.Tensor:
    """Two HxWx3 uint8 frames -> (1, T=2, C, 256, 256) normalised float."""
    x = torch.from_numpy(np.stack([prev, cur])).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(RES, RES), mode="bilinear", antialias=True, align_corners=False)
    return ((x.unsqueeze(0) - MEAN) / STD)[0]


@torch.no_grad()
def encode(model: VJEPA2Model, clips: list[torch.Tensor], device: str) -> tuple[np.ndarray, np.ndarray]:
    tokens = model.get_vision_features(torch.stack(clips).to(device))  # (B, 1*16*16, 1024)
    b, n, d = tokens.shape
    side = int(n**0.5)
    grid = tokens.view(b, side, side, d).permute(0, 3, 1, 2)  # (B, D, 16, 16)
    pooled = F.adaptive_avg_pool2d(grid, 4).flatten(2).transpose(1, 2).flatten(1)  # (B, 16*D)
    return tokens.mean(1).cpu().half().numpy(), pooled.cpu().half().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="bklassen3434/pick_pen_v2_20260920_124400")
    p.add_argument("--cam", default="top", choices=["top", "wrist"])
    p.add_argument("--stride", type=int, default=3, help="embed every Nth frame (3 = 10 per second)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit-episodes", type=int, default=None, help="quick smoke test")
    args = p.parse_args()

    root = CACHE / args.repo
    eps, data = load_meta(root)
    if args.limit_episodes:
        eps = eps[eps.episode_index < args.limit_episodes]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = VJEPA2Model.from_pretrained(MODEL_ID).to(device).eval()
    print(f"{MODEL_ID} on {device}; {args.cam} camera, {len(eps)} episodes, every {args.stride} frames")

    rows: dict[str, list] = {k: [] for k in ["episode", "frame", "mean", "grid", "pixels"]}
    pens: dict[int, tuple] = {}
    batch, batch_keys, prev = [], [], None
    t0 = time.time()

    def flush() -> None:
        m, g = encode(model, batch, device)
        rows["mean"].append(m)
        rows["grid"].append(g)
        for ep, idx in batch_keys:
            rows["episode"].append(ep)
            rows["frame"].append(idx)
        batch.clear()
        batch_keys.clear()

    for ep, idx, img in iter_frames(root, eps, args.cam):
        if idx == 0:
            prev = img
        if idx == 0 and args.cam == "top":
            pens[ep] = (find_pen_px(img, "blue"), find_pen_px(img, "pink"))
        if idx % args.stride == 0:
            batch.append(to_clip(prev, img))
            batch_keys.append((ep, idx))
            small = F.interpolate(torch.from_numpy(img).permute(2, 0, 1)[None].float(), size=(24, 32), mode="area")
            rows["pixels"].append(small[0].byte().numpy().reshape(-1))
            if len(batch) == args.batch:
                flush()
                done = len(rows["episode"])
                print(f"  {done} clips, {done / (time.time() - t0):.1f} clips/s", end="\r")
        prev = img
    if batch:
        flush()
    print(f"\nencoded {len(rows['episode'])} clips in {time.time() - t0:.0f}s")

    ep_arr, fr_arr = np.array(rows["episode"]), np.array(rows["frame"])
    OUT_DIR.mkdir(exist_ok=True)
    if args.cam != "top":  # labels live in the top-camera file; this one is just features
        out = OUT_DIR / f"embeddings_{args.cam}.npz"
        np.savez_compressed(out, episode=ep_arr, frame=fr_arr, mean=np.concatenate(rows["mean"]),
                            grid=np.concatenate(rows["grid"]), pixels=np.stack(rows["pixels"]))
        print(f"wrote {out}")
        return
    lab = data.loc[list(zip(ep_arr, fr_arr))]
    tasks = pd.read_parquet(root / "meta/tasks.parquet").reset_index()  # columns: task, task_index
    task_names = dict(zip(tasks.task_index, tasks.task))
    lengths = dict(zip(eps.episode_index, eps.length))

    bad = [ep for ep, (b, k) in pens.items() if b is None or k is None]
    if bad:
        print(f"WARNING: pen detector missed a pen in episodes {bad}; they get blue_left = -1")
    blue_left = {ep: (-1 if ep in bad else int(b[0] < k[0])) for ep, (b, k) in pens.items()}

    out = OUT_DIR / "embeddings.npz"
    np.savez_compressed(
        out,
        episode=ep_arr,
        frame=fr_arr,
        progress=fr_arr / np.array([lengths[e] for e in ep_arr]),
        state=np.stack(lab["observation.state"].to_numpy()).astype(np.float32),
        task=np.array([task_names[t] for t in lab["task_index"]]),
        blue_left=np.array([blue_left[e] for e in ep_arr]),
        mean=np.concatenate(rows["mean"]),
        grid=np.concatenate(rows["grid"]),
        pixels=np.stack(rows["pixels"]),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
