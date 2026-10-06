"""Step 2 of the V-JEPA 2 world-model experiment: an action-conditioned latent predictor.

    (V-JEPA grid embedding now, state now, the next 15 actions)  ->  V-JEPA grid embedding 0.5 s later

The encoder stays frozen; only a small transformer (~4M params) is trained, on embeddings.npz from
embed.py. It predicts the CHANGE in the embedding, so "nothing changes" is where it starts.

Scored on held-out episodes against two baselines:
  copy        "nothing changes": predict the current embedding
  no-action   the same predictor trained with the actions blanked out (can it imagine without
              being told the plan? the gap to the real model is what the actions buy)
Metrics:
  latent error      L1 error of the predicted embedding, as a fraction of the copy baseline's (lower = better)
  decoded joints    read joint angles back out of the PREDICTED embedding with a linear probe
                    (fit on real embeddings) and compare to where the arm really was 0.5 s later
  retrieval         is the real future frame the nearest neighbour of the prediction, among all
                    held-out frames? (top-1 / top-5, counting +-1 embedded step as a hit)
  counterfactual    same start frame, but the actions of a held-out episode heading to the OTHER
                    side: does the decoded future pan follow the swapped plan?
Everything is also reported on "moving" frames only (the arm moves > 5 deg in the 0.5 s), since
idle frames are trivially easy.

    cd /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1 && \
        .venv/bin/python my_contributions/vjepa/world_model.py --cams top wrist

`--cams top wrist` concatenates both cameras' 4x4 grids (32 tokens); needs `embed.py --cam wrist` first.
Outputs are tagged by camera set, e.g. outputs/world_model_top+wrist.pt.
"""

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from embed import CACHE, OUT_DIR, iter_frames, load_meta  # noqa: E402
from probe import Ridge, split_episodes  # noqa: E402

REPO = "bklassen3434/pick_pen_v2_20260920_124400"
K = 15  # frames ahead (0.5 s at 30 fps); must be a multiple of the embedding stride
STRIDE = 3
TOK_PER_CAM, D_EMB = 16, 1024
MOVING_DEG = 5.0
SEED = 0


class Predictor(nn.Module):
    def __init__(self, d_act: int, n_tok: int, d: int = 256, layers: int = 4):
        super().__init__()
        self.inp = nn.Linear(D_EMB, d)
        self.act = nn.Sequential(nn.Linear(d_act, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(1, n_tok + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, D_EMB)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.act(a)[:, None], self.inp(z)], 1) + self.pos
        return z + self.out(self.tf(h)[:, 1:])


def load_features(cams: list[str]) -> dict:
    """Labels from the top-camera file; `grid` = the chosen cameras' grids side by side."""
    raw = dict(np.load(OUT_DIR / "embeddings.npz"))
    grids = []
    for cam in cams:
        f = raw if cam == "top" else np.load(OUT_DIR / f"embeddings_{cam}.npz")
        assert (f["episode"] == raw["episode"]).all() and (f["frame"] == raw["frame"]).all(), cam
        grids.append(f["grid"])
    raw["grid"] = np.concatenate(grids, 1)
    return raw


def build_pairs(d: dict, data: pd.DataFrame) -> dict:
    """Pair each embedded frame t with t+K in the same episode, plus the action chunk a[t:t+K]."""
    key = {(e, f): i for i, (e, f) in enumerate(zip(d["episode"], d["frame"]))}
    src, dst, acts = [], [], []
    for i, (e, f) in enumerate(zip(d["episode"], d["frame"])):
        j = key.get((e, f + K))
        if j is None:
            continue
        chunk = np.stack(data.loc[(e, f) : (e, f + K - 1), "action"].to_numpy())
        src.append(i)
        dst.append(j)
        acts.append(np.concatenate([d["state"][i], chunk.reshape(-1)]))
    return {"src": np.array(src), "dst": np.array(dst), "act": np.stack(acts).astype(np.float32)}


def train(z, a, tgt, tr, va, use_actions: bool, device: str, epochs: int = 80) -> Predictor:
    torch.manual_seed(SEED)
    model = Predictor(a.shape[1], z.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=epochs * (len(tr) // 128 + 1))
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(tr)).split(128):
            b = tr[idx]
            ab = a[b] if use_actions else torch.zeros_like(a[b])
            loss = (model(z[b], ab) - tgt[b]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        val = evaluate(model, z, a, tgt, va, use_actions)
        if val < best:
            best, best_state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 19:
            print(f"    epoch {ep + 1}: val L1 {val:.4f} (best {best:.4f})")
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def predict(model, z, a, idx, use_actions: bool) -> torch.Tensor:
    out = []
    for b in idx.split(512):
        ab = a[b] if use_actions else torch.zeros_like(a[b])
        out.append(model(z[b], ab))
    return torch.cat(out)


def evaluate(model, z, a, tgt, idx, use_actions) -> float:
    model.eval()
    return (predict(model, z, a, idx, use_actions) - tgt[idx]).abs().mean().item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cams", nargs="+", default=["top"], choices=["top", "wrist"])
    args = p.parse_args()
    tag = "+".join(args.cams)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    raw = load_features(args.cams)
    n_tok = TOK_PER_CAM * len(args.cams)
    _, data = load_meta(CACHE / REPO)
    pairs = build_pairs(raw, data)
    src, dst = pairs["src"], pairs["dst"]
    episode = raw["episode"][src]
    target_left = (raw["task"] == "blue") == (raw["blue_left"] == 1)
    group = np.char.add(raw["task"].astype(str), raw["blue_left"].astype(str))[src]

    test = split_episodes(episode, group)
    val = np.zeros_like(test)
    val[~test] = split_episodes(episode[~test], group[~test])
    train_m = ~test & ~val
    print(f"{len(src)} pairs: {train_m.sum()} train / {val.sum()} val / {test.sum()} test "
          f"({len(np.unique(episode[test]))} held-out episodes)")

    # standardise embeddings (per dim, train frames only) and actions
    grid = raw["grid"].astype(np.float32)
    tr_frames = np.unique(src[train_m])
    mu, sd = grid[tr_frames].mean(0), grid[tr_frames].std(0) + 1e-4
    zs = torch.from_numpy((grid - mu) / sd).view(-1, n_tok, D_EMB)
    amu, asd = pairs["act"][train_m].mean(0), pairs["act"][train_m].std(0) + 1e-4
    acts = torch.from_numpy((pairs["act"] - amu) / asd)

    z = zs[src].to(device)
    tgt = zs[dst].to(device)
    a = acts.to(device)
    t = lambda m: torch.from_numpy(np.where(m)[0]).to(device)  # noqa: E731
    tr_i, va_i, te_i = t(train_m), t(val), t(test)

    models = {}
    for name, use in [("action", True), ("no-action", False)]:
        print(f"  training {name} predictor")
        models[name] = train(z, a, tgt, tr_i, va_i, use, device)

    preds = {
        "copy": z[te_i],
        "no-action": predict(models["no-action"], z, a, te_i, False),
        "action": predict(models["action"], z, a, te_i, True),
    }
    truth = tgt[te_i]

    # joint decoder: linear probe from real (standardised, flattened) embeddings to joint angles
    flat = zs.reshape(len(zs), -1)
    dec = Ridge().fit(flat[tr_frames], torch.from_numpy(raw["state"][tr_frames]))
    decode = lambda p: dec.predict(p.reshape(len(p), -1).cpu(), alpha=1e-2).numpy()  # noqa: E731
    true_future = raw["state"][dst[test]]
    now_state = raw["state"][src[test]]
    moving = np.abs(true_future - now_state)[:, :5].max(1) > MOVING_DEG

    # retrieval pool: every embedded frame of the held-out episodes
    pool_idx = np.where(np.isin(raw["episode"], np.unique(episode[test])))[0]
    pool = zs[pool_idx].reshape(len(pool_idx), -1).to(device)
    pool_ep, pool_fr = raw["episode"][pool_idx], raw["frame"][pool_idx]
    true_ep, true_fr = raw["episode"][dst[test]], raw["frame"][dst[test]]

    def retrieval(p: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        dist = torch.cdist(p.reshape(len(p), -1), pool)
        top = dist.topk(5, largest=False).indices.cpu().numpy()
        hit = (pool_ep[top] == true_ep[:, None]) & (np.abs(pool_fr[top] - true_fr[:, None]) <= STRIDE)
        return hit[:, 0], hit.any(1), top[:, 0]

    copy_err = (preds["copy"] - truth).abs().mean((1, 2)).cpu().numpy()
    results: dict = {}
    nn_frames = {}
    for name, p in preds.items():
        err = (p - truth).abs().mean((1, 2)).cpu().numpy()
        joints = decode(p)
        top1, top5, nn_idx = retrieval(p)
        nn_frames[name] = nn_idx
        r = {}
        for subset, m in [("all", np.ones_like(moving)), ("moving", moving)]:
            r[subset] = {
                "latent_err_vs_copy": float(err[m].mean() / copy_err[m].mean()),
                "pan_err_deg": float(np.abs(joints[m, 0] - true_future[m, 0]).mean()),
                "gripper_err_deg": float(np.abs(joints[m, 5] - true_future[m, 5]).mean()),
                "retrieval_top1": float(top1[m].mean()),
                "retrieval_top5": float(top5[m].mean()),
            }
        results[name] = r

    # counterfactual: keep this frame and this arm state, but swap in the NEXT 15 ACTIONS of a held-out
    # pair heading to the other side whose arm is in (nearly) the same pose right now (< 5 deg on every
    # joint), so the only contradiction is the plan itself. Futures must differ by > 10 deg of pan.
    te_rows = np.where(test)[0]
    st = raw["state"][src]
    cf_src, cf_act, cf_own_pan, cf_swap_pan = [], [], [], []
    for i in te_rows:
        other = te_rows[target_left[src[te_rows]] != target_left[src[i]]]
        gap_now = np.abs(st[other, :5] - st[i, :5]).max(1)
        gap_future = np.abs(raw["state"][dst[other], 0] - raw["state"][dst[i], 0])
        ok = (gap_now < 5) & (gap_future > 10)
        if ok.any():
            j = other[ok][gap_now[ok].argmin()]
            cf_src.append(i)
            cf_act.append(np.concatenate([st[i], pairs["act"][j, 6:]]))
            cf_own_pan.append(raw["state"][dst[i], 0])
            cf_swap_pan.append(raw["state"][dst[j], 0])
    cf_a = torch.from_numpy((np.stack(cf_act) - amu) / asd).to(device)
    with torch.no_grad():
        cf_pred = models["action"](z[torch.tensor(cf_src, device=device)], cf_a)
    cf_pan = decode(cf_pred)[:, 0]
    own, swap = np.array(cf_own_pan), np.array(cf_swap_pan)
    results["counterfactual"] = {
        "n": len(cf_src),
        "follows_swapped_plan": float((np.abs(cf_pan - swap) < np.abs(cf_pan - own)).mean()),
        "pan_err_to_swapped_deg": float(np.abs(cf_pan - swap).mean()),
        "own_vs_swapped_gap_deg": float(np.abs(own - swap).mean()),
    }

    print(f"\nheld-out episodes, predicting {K / 30:.1f} s ahead ({moving.sum()} of {len(moving)} test pairs are 'moving')")
    for subset in ["all", "moving"]:
        print(f"\n[{subset}]                    copy   no-action     action")
        for metric, fmt in [("latent_err_vs_copy", "{:>10.2f}"), ("pan_err_deg", "{:>10.1f}"),
                            ("gripper_err_deg", "{:>10.1f}"), ("retrieval_top1", "{:>10.0%}"),
                            ("retrieval_top5", "{:>10.0%}")]:
            print(f"  {metric:<22}" + "".join(fmt.format(results[n][subset][metric]) for n in preds))
    cf = results["counterfactual"]
    print(f"\ncounterfactual ({cf['n']} cases, own vs swapped futures differ by {cf['own_vs_swapped_gap_deg']:.0f} deg pan): "
          f"decoded future follows the SWAPPED plan {cf['follows_swapped_plan']:.0%} of the time, "
          f"{cf['pan_err_to_swapped_deg']:.1f} deg from it")
    (OUT_DIR / f"world_model_results_{tag}.json").write_text(json.dumps(results, indent=2))
    torch.save({"models": {k: m.state_dict() for k, m in models.items()}, "cams": args.cams, "n_tok": n_tok,
                "d_act": acts.shape[1], "mu": mu, "sd": sd, "amu": amu, "asd": asd,
                "test_episodes": np.unique(episode[test]), "train_frames": tr_frames},
               OUT_DIR / f"world_model_{tag}.pt")

    # picture: for a few moving test pairs -> now | real future | nearest frame to the prediction (action / no-action)
    rng = np.random.default_rng(SEED)
    show = rng.choice(np.where(moving)[0], 4, replace=False)
    want = {}
    for s in show:
        want[(raw["episode"][src[test][s]], raw["frame"][src[test][s]])] = None
        want[(true_ep[s], true_fr[s])] = None
        for n in ["action", "no-action"]:
            k = nn_frames[n][s]
            want[(pool_ep[k], pool_fr[k])] = None
    eps, _ = load_meta(CACHE / REPO)
    eps = eps[eps.episode_index.isin({e for e, _ in want})]
    for e, f, img in iter_frames(CACHE / REPO, eps):
        if (e, f) in want:
            want[(e, f)] = cv2.resize(img, (320, 240))

    def tile(e, f, text):
        img = want[(e, f)].copy()
        cv2.rectangle(img, (0, 0), (320, 22), (0, 0, 0), -1)
        cv2.putText(img, f"{text} (ep{e} f{f})", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return img

    rows = []
    for s in show:
        e0, f0 = raw["episode"][src[test][s]], raw["frame"][src[test][s]]
        ka, kn = nn_frames["action"][s], nn_frames["no-action"][s]
        rows.append(np.hstack([tile(e0, f0, "now"), tile(true_ep[s], true_fr[s], "real +0.5s"),
                               tile(pool_ep[ka], pool_fr[ka], "imagined: action"),
                               tile(pool_ep[kn], pool_fr[kn], "imagined: no-action")]))
    out = OUT_DIR / f"world_model_examples_{tag}.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
