"""A language-conditioned progress model: "how far along is this frame, toward THAT pen?"

Given one frame (top + wrist camera) and a colour word, it returns a number in [0, 1]:
0 = nothing has happened, 1 = the named pen has been picked up.

Why bother, given the policy is the thing being trained? Because this project had no
automated way to tell whether a rollout worked -- every evaluation was either an offline
action-delta probe or a human watching the arm. This gives a *curve* per rollout: did it
reach 1.0, and if not, where did it stall.

The language conditioning is the part that matters. A model trained only on "is a pen in the
gripper" would score this project's known failure mode -- always grabbing the same pen -- as
a perfect success. So every frame is scored for every colour, and the wrong colour is a
negative. That makes it a success detector for picking the *right* pen.

TWO KINDS OF TRAINING DATA, and the second is what makes it trustworthy:

  demonstrations (`embed <repo>`)
      Labels come from `progress_labels.py`, which reads the events out of the action
      stream, so nothing is hand-annotated. Every demo succeeds, which is the problem:
      a model trained on these alone scores "the gripper closed next to the pink pen" at
      0.90 even when the pen never left the table.

  rollouts (`embed <repo> --annotations my_contributions/rollout_outcomes.json`)
      Real attempts, mostly failures. Which pen the gripper closed on -- and whether it
      actually came up -- cannot be read from the joints (one episode lifted a pen with 1.1
      deg of shoulder_lift; the lift came from the wrist), so those outcomes are checked by
      eye once and recorded in rollout_outcomes.json. Frames where the arm reached, closed,
      and failed are the only examples of a *failed* grasp that exist anywhere.

      Adding 13 rollout episodes to 60 demos left every demonstration metric untouched and
      dropped the score on a failed grasp from 0.84 to 0.51, and on episodes where the arm
      never reached a pen at all from 0.21 to 0.00. Held-out failures went 0.90/0.88 ->
      0.49/0.53.

Architecture is deliberately small: a frozen SigLIP vision tower (the family SmolVLA uses)
with a 2-layer MLP head. The backbone never trains, so its features are cached once and the
head then trains in a minute on a laptop. Head inputs are the top and wrist embeddings plus
their difference against a frame 0.5 s earlier -- without that delta a single frame cannot
tell reaching-down from lifting-up.

Two losses: a masked soft-target BCE over every (frame, colour) pair, and a pairwise ranking
hinge on frames from the same episode (later must score higher). The ranking term is what
keeps the curve smooth and robust to operators moving at different speeds; it is applied to
demonstrations only, because a rollout can approach, miss, retreat and try again.

Usage (all from /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1):

    # 1. cache SigLIP features  (~6 min for 60 demo episodes on an M3 Pro)
    .venv/bin/python -m my_contributions.tools.progress_model embed \
        bklassen3434/pick_pen_v2_20260920_124400
    .venv/bin/python -m my_contributions.tools.progress_model embed \
        bklassen3434/rollout_pen_20260925_155645 \
        --annotations my_contributions/rollout_outcomes.json

    # 2. train the head on all of it  (~1 min, no GPU needed)
    uv run --no-sync --with matplotlib python -m my_contributions.tools.progress_model train \
        bklassen3434/pick_pen_v2_20260920_124400 bklassen3434/rollout_pen_20260925_155645

    # 3. grade any dataset -- no labels needed, so it works on failures
    uv run --no-sync --with matplotlib python -m my_contributions.tools.progress_model score \
        <repo_id> --all-tasks

`lerobot-rollout` writes a normal LeRobot dataset (see rollout_eval.sh), so step 3 turns a
policy evaluation into a table: what it was told, where the gripper closed, and which colour
the model thinks it picked.

Read the *peak* of a rollout curve, not its final value: training stops ~1.5 s after the
grasp, so a long hold drifts out of distribution. And before trusting the `grasped` column,
render `progress_labels.py --grasp-frames` -- it is a heuristic, and the frames are not.

Matplotlib is not in the project venv, so the plotting paths need:
    uv run --no-sync --with matplotlib python -m my_contributions.tools.progress_model ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from my_contributions.tools.progress_labels import events, frame_labels, rollout_labels

CACHE = Path.home() / ".cache/huggingface/lerobot"
OUT = Path(__file__).resolve().parents[2] / "outputs/progress_model"
BACKBONE = "google/siglip-base-patch16-224"
EMB_DIM = 768
DELTA_FRAMES = 15  # 0.5 s back, for the motion-direction feature
VAL_EVERY = 5  # every 5th episode is held out


# --------------------------------------------------------------------------- embedding


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def _tag(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def dataset_tasks(repo_id: str) -> list[str]:
    """Sorted colour vocabulary of a dataset, read from its metadata."""
    t = pd.read_parquet(CACHE / repo_id / "meta/tasks.parquet").reset_index()
    return sorted(str(x) for x in t["task"])


def embed(repo_id: str, batch: int = 32, overwrite: bool = False, annotations: str | None = None) -> Path:
    """Cache frozen SigLIP features for every labelled frame of a dataset.

    Without --annotations this is a demonstration dataset: one label per frame (its own
    colour) and every other colour is a negative, derived at train time.

    With --annotations it is a *rollout*, and the per-(frame, colour) targets come from
    `rollout_labels`. Those are stored explicitly as target/mask matrices, because a rollout
    frame can be a hard negative for one colour and unlabelled for the other.
    """
    from transformers import AutoModel

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = OUT / f"{_tag(repo_id)}_siglip.npz"
    if out.exists() and not overwrite:
        print(f"{out} exists; pass --overwrite to rebuild")
        return out

    tasks = dataset_tasks(repo_id)
    ev = events(repo_id)
    target = mask = None
    if annotations:
        outcomes = json.loads(Path(annotations).read_text())[repo_id]["episodes"]
        long = rollout_labels(repo_id, outcomes, tasks)
        labels = long[["ep", "frame"]].drop_duplicates().sort_values(["ep", "frame"]).reset_index(drop=True)
        labels["task"] = tasks[0]  # placeholder; the real targets live in the matrices
        labels["progress"] = 0.0
        pos = {(e, f): i for i, (e, f) in enumerate(zip(labels.ep, labels.frame, strict=True))}
        target = np.zeros((len(labels), len(tasks)), np.float32)
        mask = np.zeros((len(labels), len(tasks)), bool)
        for row in long.itertuples():
            i, k = pos[(row.ep, row.frame)], tasks.index(row.task)
            target[i, k] = row.target
            mask[i, k] = True
        print(f"rollout: {len(labels)} frames, {mask.sum()} labelled (frame, colour) pairs")
    else:
        labels = frame_labels(repo_id, ev)
    print(f"{len(labels)} frames to embed from {labels.ep.nunique()} episodes")

    ds = LeRobotDataset(repo_id, root=str(CACHE / repo_id))
    starts = {int(e): int(s) for e, s in enumerate(ds.meta.episodes["dataset_from_index"])}
    cams = ["observation.images.top", "observation.images.wrist"]

    dev = _device()
    model = AutoModel.from_pretrained(BACKBONE, dtype=torch.float32).eval().to(dev)

    feats = {c: np.zeros((len(labels), EMB_DIM), dtype=np.float16) for c in cams}
    rows = labels.itertuples()
    buf: dict[str, list[torch.Tensor]] = {c: [] for c in cams}
    idx: list[int] = []

    def flush() -> None:
        if not idx:
            return
        with torch.no_grad():
            for c in cams:
                x = torch.stack(buf[c]).to(dev)
                f = model.get_image_features(pixel_values=x).pooler_output
                f = F.normalize(f.float(), dim=-1)  # keeps head inputs on a fixed scale
                feats[c][idx] = f.cpu().numpy().astype(np.float16)
        idx.clear()
        for c in cams:
            buf[c].clear()

    for i, r in enumerate(rows):
        item = ds[starts[r.ep] + r.frame]
        for c in cams:
            # LeRobotDataset hands back CHW float in [0,1]; SigLIP wants 224x224 at mean/std 0.5.
            img = F.interpolate(item[c][None], size=(224, 224), mode="bilinear", antialias=True)[0]
            buf[c].append((img - 0.5) / 0.5)
        idx.append(i)
        if len(idx) == batch:
            flush()
            if (i + 1) % (batch * 20) == 0:
                print(f"  {i + 1}/{len(labels)}", flush=True)
    flush()

    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if target is None else {"target": target, "mask": mask}
    np.savez_compressed(
        out,
        **extra,
        top=feats[cams[0]],
        wrist=feats[cams[1]],
        ep=labels.ep.values.astype(np.int32),
        frame=labels.frame.values.astype(np.int32),
        progress=labels.progress.values.astype(np.float32),
        task_id=labels.task.map({t: i for i, t in enumerate(tasks)}).values.astype(np.int64),
        tasks=np.array(tasks),
    )
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    return out


# --------------------------------------------------------------------------- model


class ProgressHead(nn.Module):
    """[top, wrist, top-delta, wrist-delta, task] -> progress in [0, 1]."""

    def __init__(self, n_tasks: int, task_dim: int = 64, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.task_emb = nn.Embedding(n_tasks, task_dim)
        d = 4 * EMB_DIM + task_dim
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.task_emb(task_id)], dim=-1)).squeeze(-1)


def build_features(z: dict, ep: np.ndarray) -> np.ndarray:
    """Concatenate the two views with their 0.5 s deltas.

    Kept frames are contiguous within an episode, so "DELTA_FRAMES earlier" is just an offset
    into the episode's slice, clamped at its start.
    """
    top, wrist = z["top"].astype(np.float32), z["wrist"].astype(np.float32)
    prev = np.arange(len(ep))
    for e in np.unique(ep):
        sel = np.where(ep == e)[0]
        prev[sel] = sel[np.maximum(0, np.arange(len(sel)) - DELTA_FRAMES)]
    return np.concatenate([top, wrist, top - top[prev], wrist - wrist[prev]], axis=1)


# --------------------------------------------------------------------------- training


def _split(ep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    val_eps = np.unique(ep)[::VAL_EVERY]
    is_val = np.isin(ep, val_eps)
    return ~is_val, is_val


def _rank_pairs(
    ep: np.ndarray, rng: np.random.Generator, n: int, by_ep: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """n random (a, b) index pairs drawn from the same episode."""
    by_ep = by_ep if by_ep is not None else {e: np.where(ep == e)[0] for e in np.unique(ep)}
    keys = list(by_ep)
    eps = rng.choice(len(keys), size=n)
    a = np.array([rng.choice(by_ep[keys[e]]) for e in eps])
    b = np.array([rng.choice(by_ep[keys[e]]) for e in eps])
    return a, b


def load_cache(repo_id: str, tasks: list[str] | None = None) -> dict:
    """One cached dataset as features + (target, mask) matrices over the task vocabulary.

    Demonstration caches store one label per frame; the matrices are derived from the
    convention that the episode's own colour carries the ramp and every other colour is a
    zero. Rollout caches store the matrices directly, because their targets are not
    derivable -- they came from hand-checked outcomes.
    """
    z = np.load(OUT / f"{_tag(repo_id)}_siglip.npz", allow_pickle=False)
    own = [str(t) for t in z["tasks"]]
    tasks = tasks or own
    if own != tasks:
        raise SystemExit(f"task vocabulary mismatch: {repo_id} has {own}, expected {tasks}")

    ep, x = z["ep"], build_features(z, z["ep"])
    n, k = len(ep), len(tasks)
    if "target" in z.files:
        target, mask = z["target"].astype(np.float32), z["mask"].astype(bool)
        is_demo = False
    else:
        target = np.zeros((n, k), np.float32)
        mask = np.ones((n, k), bool)
        target[np.arange(n), z["task_id"]] = z["progress"]
        is_demo = True
    return {
        "repo": repo_id,
        "x": x,
        "ep": ep,
        "target": target,
        "mask": mask,
        "own_progress": z["progress"],
        "own_task": z["task_id"],
        "is_demo": is_demo,
        "tasks": tasks,
    }


def train(
    repos: list[str],
    epochs: int = 60,
    batch: int = 256,
    lr: float = 1e-3,
    rank_weight: float = 0.5,
    rank_margin: float = 0.05,
    rollout_weight: float = 3.0,
    seed: int = 0,
) -> Path:
    """Train the head on one or more cached datasets.

    Rollout frames are upweighted: there are an order of magnitude fewer of them than
    demonstration frames, but they carry the only examples of a grasp that *failed*, which is
    the thing demonstrations structurally cannot teach.
    """
    caches = [load_cache(repos[0])]
    tasks = caches[0]["tasks"]
    caches += [load_cache(r, tasks) for r in repos[1:]]

    # Episode ids repeat across datasets; offset them so ranking pairs and the val split never
    # mix two different recordings into one "episode".
    offset, eps = 0, []
    for c in caches:
        eps.append(c["ep"] + offset)
        offset += int(c["ep"].max()) + 1
    ep = np.concatenate(eps)
    x = np.concatenate([c["x"] for c in caches])
    target = np.concatenate([c["target"] for c in caches])
    mask = np.concatenate([c["mask"] for c in caches])
    demo = np.concatenate([np.full(len(c["ep"]), c["is_demo"]) for c in caches])
    weight = np.where(demo, 1.0, rollout_weight).astype(np.float32)
    own_progress = np.concatenate([c["own_progress"] for c in caches])
    own_task = np.concatenate([c["own_task"] for c in caches])

    for c, r in zip(caches, repos, strict=True):
        kind = "demo" if c["is_demo"] else "rollout"
        print(f"  {r}: {len(c['ep'])} frames, {len(np.unique(c['ep']))} eps ({kind})")

    tr, va = _split(ep)
    print(f"train {tr.sum()} frames / {len(np.unique(ep[tr]))} eps | val {va.sum()} / {len(np.unique(ep[va]))}")

    dev = _device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ProgressHead(len(tasks)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    feat = torch.from_numpy(x).to(dev)
    targ = torch.from_numpy(target).to(dev)
    msk = torch.from_numpy(mask.astype(np.float32)).to(dev)
    wgt = torch.from_numpy(weight).to(dev)
    all_tids = torch.arange(len(tasks), device=dev)

    tr_idx = np.where(tr)[0]
    # Ranking only makes sense where progress is known to be monotone in time, i.e. on
    # demonstrations. A rollout can approach, miss, retreat and try again.
    rank_idx = np.where(tr & demo)[0]
    rank_by_ep = {e: np.where(ep[rank_idx] == e)[0] for e in np.unique(ep[rank_idx])}

    m: dict = {}
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        tot = 0.0
        for s in range(0, len(perm), batch):
            b = torch.from_numpy(perm[s : s + batch]).to(dev)

            # Every colour is scored for every frame; the mask says which pairs are supervised.
            preds = torch.stack([model(feat[b], all_tids[k].expand(len(b))) for k in range(len(tasks))], dim=1)
            bce = F.binary_cross_entropy_with_logits(preds, targ[b], reduction="none")
            w = msk[b] * wgt[b][:, None]
            loss = (bce * w).sum() / w.sum().clamp(min=1)

            # Ranking: later frames of the same demo episode must score higher. Carries the
            # ordering even where the regression target and the image disagree (the arm
            # hovers over the pen for ~1 s while the linear label keeps climbing).
            if rank_weight:
                ja, jb = _rank_pairs(None, rng, len(b), rank_by_ep)
                ja, jb = rank_idx[ja], rank_idx[jb]
                ta, tb = torch.from_numpy(ja).to(dev), torch.from_numpy(jb).to(dev)
                ka = torch.from_numpy(own_task[ja]).to(dev)
                kb = torch.from_numpy(own_task[jb]).to(dev)
                pa, pb = model(feat[ta], ka), model(feat[tb], kb)
                ya, yb = own_progress[ja], own_progress[jb]
                later = torch.from_numpy((yb > ya + 0.1).astype(np.float32)).to(dev)
                earlier = torch.from_numpy((ya > yb + 0.1).astype(np.float32)).to(dev)
                gap = torch.sigmoid(pb) - torch.sigmoid(pa)
                hinge = later * F.relu(rank_margin - gap) + earlier * F.relu(rank_margin + gap)
                loss = loss + rank_weight * hinge.sum() / (later + earlier).sum().clamp(min=1)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()

        if epoch % 10 == 9 or epoch == epochs - 1:
            m = evaluate(model, feat, ep, target, mask, own_progress, own_task, demo, va, len(tasks))
            print(
                f"  epoch {epoch + 1:3d}  loss {tot / len(perm):.4f}  "
                f"demo MAE {m['demo_mae']:.3f}  order {m['order']:.3f}  "
                f"colour-gap {m['colour_gap']:+.3f} ({m['colour_correct']:.0%})  "
                f"rollout MAE {m['rollout_mae']:.3f}",
                flush=True,
            )

    ckpt = OUT / "progress_head.pt"
    torch.save({"state": model.state_dict(), "tasks": tasks, "repos": repos}, ckpt)
    (OUT / "progress_metrics.json").write_text(json.dumps(m, indent=2))
    print(f"\nwrote {ckpt}")
    plot_val(model, feat, ep, own_progress, own_task, demo, va, tasks, OUT / "progress_val.png")
    return ckpt


@torch.no_grad()
def plot_val(model, feat, ep, own_progress, own_task, demo, mask, tasks, out: Path, n: int = 6) -> None:
    """Held-out demo curves, asked with every colour -- the one plot worth looking at.

    In a blue episode the blue trace should climb to 1 and the pink trace should stay flat.
    Two traces climbing together means the model found the pen but not the instruction.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    val_eps = np.unique(ep[mask & demo])[:n]
    colours = {"blue": "tab:blue", "pink": "tab:pink", "grey": "tab:grey"}
    fig, axes = plt.subplots(len(val_eps), 1, figsize=(9, 1.9 * len(val_eps)), sharex=True)
    for ax, e in zip(np.atleast_1d(axes), val_eps, strict=False):
        sel = np.where(ep == e)[0]
        i = torch.from_numpy(sel).to(feat.device)
        ax.plot(own_progress[sel], color="k", lw=2.5, alpha=0.35, label="label")
        for k, name in enumerate(tasks):
            tk = torch.full((len(sel),), k, device=feat.device, dtype=torch.long)
            p = torch.sigmoid(model(feat[i], tk)).cpu().numpy()
            ax.plot(p, lw=1.6, color=colours.get(name, f"C{k}"), label=f'asked "{name}"')
        ax.set_ylabel(f"ep {e}\ntrue: {tasks[own_task[sel][0]]}", fontsize=8)
        ax.set_ylim(-0.05, 1.1)
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=4, loc="upper left")
    np.atleast_1d(axes)[-1].set_xlabel("frame within the kept window")
    fig.suptitle("held-out episodes: predicted progress for each colour prompt", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


@torch.no_grad()
def evaluate(model, feat, ep, target, mask, own_progress, own_task, demo, val, n_tasks) -> dict:
    model.eval()
    idx = np.where(val)[0]
    i = torch.from_numpy(idx).to(feat.device)
    preds = torch.stack(
        [torch.sigmoid(model(feat[i], torch.full((len(i),), k, device=feat.device, dtype=torch.long)))
         for k in range(n_tasks)],
        dim=1,
    ).cpu().numpy()

    def masked_mae(sub):
        mm = mask[idx][sub]
        return float(np.abs(preds[sub] - target[idx][sub])[mm].mean()) if mm.any() else float("nan")

    d = demo[idx]
    res = {"demo_mae": masked_mae(d), "rollout_mae": masked_mae(~d), "n": len(idx)}

    # Ordering, on demos only: of random within-episode pairs genuinely apart in time, how
    # many does the model rank the right way round?
    di = idx[d]
    p_own = preds[d, own_task[di]]
    y_own = own_progress[di]
    rng = np.random.default_rng(1)
    a, b = _rank_pairs(ep[di], rng, 5000)
    keep = np.abs(y_own[a] - y_own[b]) > 0.1
    res["order"] = float((np.sign(p_own[a] - p_own[b]) == np.sign(y_own[a] - y_own[b]))[keep].mean())

    # The colour test: on demo frames where the pen is already in the gripper, does naming the
    # right colour score higher than naming the wrong one?
    res["colour_gap"] = res["colour_correct"] = float("nan")
    if n_tasks > 1:
        done = y_own > 0.85
        pw = preds[d][done][np.arange(done.sum()), (own_task[di][done] + 1) % n_tasks]
        res["colour_gap"] = float((p_own[done] - pw).mean())
        res["colour_correct"] = float((p_own[done] > pw).mean())
    return res

# --------------------------------------------------------------------------- scoring


@torch.no_grad()
def score(
    repo_id: str,
    ckpt_repo: str,
    task: str | None,
    episodes: list[int] | None = None,
    all_tasks: bool = False,
    plot: bool = True,
) -> pd.DataFrame:
    """Progress curve for every episode of a dataset. No labels needed -- works on failures.

    With all_tasks, every episode is scored under every colour prompt and cross-checked
    against the grasp read out of the actions. That is the verdict table for a rollout: what
    the policy was told, where its gripper actually closed, and which colour the progress
    model thinks it picked.
    """
    from transformers import AutoModel

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ckpt = OUT / "progress_head.pt" if ckpt_repo in (None, "", "default") else Path(ckpt_repo)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    tasks = blob["tasks"]
    dev = _device()
    head = ProgressHead(len(tasks)).to(dev)
    head.load_state_dict(blob["state"])
    head.eval()
    backbone = AutoModel.from_pretrained(BACKBONE, dtype=torch.float32).eval().to(dev)

    ds = LeRobotDataset(repo_id, root=str(CACHE / repo_id))
    cams = ["observation.images.top", "observation.images.wrist"]
    # Episode bounds come from the metadata, not from touching every frame -- indexing the
    # dataset decodes video, so building the map that way costs as much as the scoring.
    bounds = {
        int(e): (int(lo), int(hi))
        for e, (lo, hi) in enumerate(
            zip(ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"], strict=True)
        )
    }
    wanted = episodes if episodes else sorted(bounds)

    rows = []
    for e in wanted:
        lo, hi = bounds[e]
        sel = np.arange(lo, hi)
        feats = {c: [] for c in cams}
        for s in range(0, len(sel), 32):
            chunk = sel[s : s + 32]
            items = [ds[int(i)] for i in chunk]
            for c in cams:
                x = torch.stack(
                    [
                        (F.interpolate(it[c][None], size=(224, 224), mode="bilinear", antialias=True)[0] - 0.5) / 0.5
                        for it in items
                    ]
                ).to(dev)
                feats[c].append(F.normalize(backbone.get_image_features(pixel_values=x).pooler_output, dim=-1))
        top = torch.cat(feats[cams[0]])
        wrist = torch.cat(feats[cams[1]])
        prev = torch.clamp(torch.arange(len(top), device=dev) - DELTA_FRAMES, min=0)
        feat = torch.cat([top, wrist, top - top[prev], wrist - wrist[prev]], dim=1)

        asked = task or ds[int(sel[0])]["task"]
        if asked not in tasks:
            raise SystemExit(f"task {asked!r} not in the model's vocabulary {tasks}")

        curves = {}
        for name in tasks if all_tasks else [asked]:
            tid = torch.full((len(feat),), tasks.index(name), device=dev, dtype=torch.long)
            curves[name] = torch.sigmoid(head(feat, tid)).cpu().numpy()

        p = curves[asked]
        reached = np.where(p > 0.9)[0]
        row = {
            "ep": int(e),
            "asked": asked,
            "n": len(p),
            "peak": float(p.max()),
            "first_at_0.9": int(reached[0]) if len(reached) else -1,
            "curve": p,
        }
        if all_tasks:
            for name, c in curves.items():
                row[f"peak_{name}"] = float(c.max())
            row["says"] = max(curves, key=lambda k: curves[k].max())
        rows.append(row)
        print(f"  ep {e:3d} [{asked}] peak {p.max():.2f}  reaches 0.9 at frame {row['first_at_0.9']}", flush=True)

    r = pd.DataFrame(rows)
    print(f"\n{len(r)} episodes | peak progress mean {r.peak.mean():.2f} | reached 0.9: {(r['first_at_0.9'] >= 0).mean():.0%}")

    if all_tasks:
        # Cross-check against the actions: where the gripper actually closed says which slot
        # the arm went to, independently of anything the progress model believes.
        ev = events(repo_id).set_index("ep")
        r["grasp_pan"] = [
            round(ev.pan_at_grasp.get(e, float("nan")), 1) if e in ev.index else float("nan") for e in r.ep
        ]
        r["grasped"] = [e in ev.index and ev.t_grasp.get(e, -1) > 0 for e in r.ep]
        cols = ["ep", "asked", "grasped", "grasp_pan", *[c for c in r.columns if c.startswith("peak_")], "says"]
        print("\nverdict table:")
        print(r[cols].to_string(index=False))
        agree = (r.asked == r.says) & r.grasped
        print(
            f"\n  grasped something : {r.grasped.sum()}/{len(r)}"
            f"\n  progress model says it picked the pen it was asked for : {agree.sum()}/{len(r)}"
            f"\n  (grasp_pan near +10 = left slot, near -33 = right slot)"
        )

    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        for row in r.itertuples():
            ax.plot(row.curve, lw=1, alpha=0.7, label=f"ep{row.ep}" if len(r) <= 12 else None)
        ax.axhline(0.9, color="k", ls=":", lw=1)
        ax.set(xlabel="frame", ylabel="predicted progress", ylim=(-0.05, 1.05), title=f"progress -- {repo_id}")
        if len(r) <= 12:
            ax.legend(fontsize=7, ncol=4)
        out = OUT / f"{_tag(repo_id)}_score.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")

    return r


# --------------------------------------------------------------------------- cli


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="cache frozen SigLIP features for the labelled frames")
    e.add_argument("repo_id")
    e.add_argument("--batch", type=int, default=32)
    e.add_argument("--overwrite", action="store_true")
    e.add_argument("--annotations", default=None, help="rollout_outcomes.json, for rollout datasets")

    t = sub.add_parser("train", help="train the progress head on one or more cached datasets")
    t.add_argument("repo_ids", nargs="+", help="demo dataset first, then any rollouts")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--rank-weight", type=float, default=0.5)
    t.add_argument("--rollout-weight", type=float, default=3.0)

    s = sub.add_parser("score", help="progress curve per episode of any dataset")
    s.add_argument("repo_id")
    s.add_argument("--ckpt", default="default", help="path to a head checkpoint; default is the latest")
    s.add_argument("--task", default=None, help="colour word; default is each episode's own task")
    s.add_argument("--episodes", type=int, nargs="+", default=None, help="only these episode indices")
    s.add_argument("--all-tasks", action="store_true", help="score every prompt and print the verdict table")

    a = p.parse_args()
    if a.cmd == "embed":
        embed(a.repo_id, a.batch, a.overwrite, a.annotations)
    elif a.cmd == "train":
        train(a.repo_ids, epochs=a.epochs, lr=a.lr, rank_weight=a.rank_weight,
              rollout_weight=a.rollout_weight)
    else:
        score(a.repo_id, a.ckpt, a.task, a.episodes, a.all_tasks)


if __name__ == "__main__":
    main()
