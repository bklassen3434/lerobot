"""Step 3 of the V-JEPA 2 world-model experiment: plan by imagining, toward a goal picture.

Given the current camera view and a GOAL picture, search for the arm motion whose imagined future
(world_model.py's predictor, rolled out 0.5 s at a time) looks most like the goal, in V-JEPA space.
No policy, no reward, no training. Just "try plans in your head, keep what lands closest".

Search: the cross-entropy method (CEM). A plan = one joint-angle waypoint per 0.5 s, the 15 actions
in between linearly interpolated. Sample 1024 plans, imagine each, keep the best 64, re-centre the
sampling on them, repeat 12 times.

Test, on the 13 held-out episodes, starting where the arm begins to move:
  own goal      the goal is this episode's own frame H*0.5 s later
  swapped goal  the goal is a frame from a held-out episode with the SAME pen layout but the OTHER
                target pen. Same start picture, different goal: does the plan go the other way?
Scored by the planned final shoulder_pan vs the pan actually seen in the goal frame, and by "right
pen": does the plan end nearer the goal's pan than the pan of the alternative goal from the same start
(the other target pen)? Both pens lie the same direction from the rest pose, so "moved toward the
goal" would be meaningless. Baselines: "stay still", "random plan", and the real demo actions.

    cd /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1 && \
        .venv/bin/python my_contributions/vjepa/plan.py --cams top wrist --horizon 3
"""

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from embed import CACHE, OUT_DIR, iter_frames, load_meta  # noqa: E402
from world_model import D_EMB, K, REPO, STRIDE, Predictor, load_features  # noqa: E402

ONSET_DEG = 3.0  # "arm starts moving" = any joint 3 deg from its starting pose
INIT_STD = torch.tensor([20.0, 20.0, 20.0, 20.0, 20.0, 10.0])  # CEM starting spread, deg per joint
MIN_STD = 1.0


class Planner:
    def __init__(self, ckpt: dict, device: str, lo: np.ndarray, hi: np.ndarray):
        self.model = Predictor(ckpt["d_act"], ckpt["n_tok"]).to(device).eval()
        self.model.load_state_dict(ckpt["models"]["action"])
        self.amu = torch.from_numpy(ckpt["amu"]).to(device)
        self.asd = torch.from_numpy(ckpt["asd"]).to(device)
        self.lo, self.hi = torch.from_numpy(lo).to(device), torch.from_numpy(hi).to(device)
        self.device = device
        self.ramp = torch.arange(1, K + 1, device=device).float().view(1, K, 1) / K

    @torch.no_grad()
    def imagine(self, z0: torch.Tensor, s0: torch.Tensor, waypoints: torch.Tensor) -> torch.Tensor:
        """z0 (T, D), s0 (6,), waypoints (N, H, 6) -> imagined final embedding (N, T, D)."""
        n = len(waypoints)
        z, s = z0.expand(n, -1, -1), s0.expand(n, -1)
        for c in range(waypoints.shape[1]):
            wp = waypoints[:, c]
            chunk = s[:, None] + (wp - s)[:, None] * self.ramp  # (N, 15, 6) straight line to the waypoint
            act = (torch.cat([s, chunk.flatten(1)], 1) - self.amu) / self.asd
            z, s = self.model(z, act), wp
        return z

    def cost(self, z0, s0, zg, waypoints) -> torch.Tensor:
        return (self.imagine(z0, s0, waypoints) - zg).abs().mean((1, 2))

    def plan(self, z0, s0, zg, horizon: int, n=1024, elites=64, iters=12, seed=0) -> tuple[torch.Tensor, float]:
        g = torch.Generator(device="cpu").manual_seed(seed)
        mean = s0.expand(horizon, -1).clone()
        std = INIT_STD.to(self.device).expand(horizon, -1).clone()
        for _ in range(iters):
            noise = torch.randn(n, horizon, 6, generator=g).to(self.device)
            samples = torch.maximum(torch.minimum(mean + std * noise, self.hi), self.lo)
            samples[0] = mean  # keep the current best guess in the running
            best = samples[self.cost(z0, s0, zg, samples).argsort()[:elites]]
            mean, std = best.mean(0), best.std(0).clamp(min=MIN_STD)
        return mean, self.cost(z0, s0, zg, mean[None]).item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cams", nargs="+", default=["top", "wrist"], choices=["top", "wrist"])
    p.add_argument("--horizon", type=int, default=3, help="number of 0.5 s steps to plan")
    args = p.parse_args()
    tag = "+".join(args.cams)
    H = args.horizon
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(OUT_DIR / f"world_model_{tag}.pt", weights_only=False)
    raw = load_features(args.cams)
    zs = torch.from_numpy((raw["grid"].astype(np.float32) - ckpt["mu"]) / ckpt["sd"]).view(-1, ckpt["n_tok"], D_EMB)
    state = raw["state"]
    planner = Planner(ckpt, device, state.min(0), state.max(0))
    _, data = load_meta(CACHE / REPO)

    key = {(e, f): i for i, (e, f) in enumerate(zip(raw["episode"], raw["frame"]))}
    target_left = (raw["task"] == "blue") == (raw["blue_left"] == 1)
    starts = {}
    for e in ckpt["test_episodes"]:
        m = np.where(raw["episode"] == e)[0]
        s = state[m]
        onset = m[np.argmax(np.abs(s[:, :5] - s[0, :5]).max(1) > ONSET_DEG)]
        f0 = raw["frame"][onset] - STRIDE  # one step before visible motion
        goal = key.get((e, f0 + H * K))
        if goal is not None:
            starts[e] = (key[(e, f0)], goal)

    cases = []  # (start index, goal index, kind)
    for e, (i, g) in starts.items():
        cases.append((i, g, "own"))
        for e2, (_, g2) in starts.items():
            if raw["blue_left"][g2] == raw["blue_left"][i] and target_left[g2] != target_left[i]:
                cases.append((i, g2, "swapped"))

    print(f"world model {tag}, planning {H * 0.5:.1f} s ahead from motion onset; "
          f"{sum(c[2] == 'own' for c in cases)} own-goal and {sum(c[2] == 'swapped' for c in cases)} swapped-goal cases")
    rows = []
    gen = torch.Generator().manual_seed(1)
    for n, (i, g, kind) in enumerate(cases):
        z0, zg = zs[i].to(device), zs[g].to(device)
        s0 = torch.from_numpy(state[i]).to(device)
        wps, cost = planner.plan(z0, s0, zg, H, seed=n)
        rand = s0 + INIT_STD.to(device) * torch.randn(64, H, 6, generator=gen).to(device)
        e, f = raw["episode"][i], raw["frame"][i]
        demo = np.stack(data.loc[(e, f) : (e, f + H * K - 1), "action"].to_numpy())[K - 1 :: K]
        demo_cost = planner.cost(z0, s0, zg, torch.from_numpy(demo).float().to(device)[None]).item() if kind == "own" else None
        rows.append({
            "kind": kind, "episode": int(e), "goal_episode": int(raw["episode"][g]), "start": i, "goal": g,
            "pan_start": float(state[i, 0]), "pan_goal": float(state[g, 0]),
            "pan_plan": float(wps[-1, 0]), "pan_random": rand[:, -1, 0].cpu().numpy().tolist(),
            "cost_plan": cost, "cost_stay": planner.cost(z0, s0, zg, s0.expand(1, H, -1)).item(),
            "cost_demo": demo_cost, "final_latent": planner.imagine(z0, s0, wps[None])[0].cpu(),
        })
        print(f"  {n + 1}/{len(cases)} {kind:<8} ep{e}->ep{raw['episode'][g]}: pan start {state[i, 0]:6.1f} "
              f"goal {state[g, 0]:6.1f} plan {wps[-1, 0]:6.1f}")

    results = {}
    for r in rows:  # the alternative: mean goal pan of the other kind of goal from the same start
        r["pan_other"] = float(np.mean([o["pan_goal"] for o in rows if o["start"] == r["start"] and o["kind"] != r["kind"]]))
    for kind in ["own", "swapped"]:
        rs = [r for r in rows if r["kind"] == kind]
        goal = np.array([r["pan_goal"] for r in rs])
        other = np.array([r["pan_other"] for r in rs])
        start = np.array([r["pan_start"] for r in rs])
        plan_ = np.array([r["pan_plan"] for r in rs])
        rand = np.array([r["pan_random"] for r in rs])  # (cases, 64)
        right = lambda x: np.abs(x - goal) < np.abs(x - other)  # noqa: E731
        results[kind] = {
            "n": len(rs),
            "pan_err_stay": float(np.abs(start - goal).mean()),
            "pan_err_random": float(np.abs(rand - goal[:, None]).mean()),
            "pan_err_plan": float(np.abs(plan_ - goal).mean()),
            "right_pen_random": float(np.mean([right(rand[:, k]).mean() for k in range(rand.shape[1])])),
            "right_pen_plan": float(right(plan_).mean()),
            "cost_stay": float(np.mean([r["cost_stay"] for r in rs])),
            "cost_plan": float(np.mean([r["cost_plan"] for r in rs])),
        }
        if kind == "own":
            results[kind]["cost_demo"] = float(np.mean([r["cost_demo"] for r in rs]))

    print(f"\nplanning {H * 0.5:.1f} s ahead       stay still   random plan   CEM plan")
    for kind in ["own", "swapped"]:
        r = results[kind]
        print(f"[{kind} goal, {r['n']} cases]")
        print(f"  pan error vs goal (deg)   {r['pan_err_stay']:10.1f}  {r['pan_err_random']:12.1f}  {r['pan_err_plan']:9.1f}")
        print(f"  ends nearer the right pen {'—':>10}  {r['right_pen_random']:12.0%}  {r['right_pen_plan']:9.0%}")
        extra = f"   (real demo actions: {r['cost_demo']:.3f})" if kind == "own" else ""
        print(f"  imagined distance to goal {r['cost_stay']:10.3f}  {'':>12}  {r['cost_plan']:9.3f}{extra}")
    (OUT_DIR / f"plan_results_{tag}_h{H}.json").write_text(json.dumps(results, indent=2))

    # picture: start | goal | nearest real frame to the imagined end of the plan, own + swapped for 3 starts
    pool_idx = np.where(np.isin(raw["episode"], ckpt["test_episodes"]))[0]
    pool = zs[pool_idx].flatten(1)
    picks = []
    for e in list(starts)[:3]:
        for kind in ["own", "swapped"]:
            r = next((r for r in rows if r["episode"] == e and r["kind"] == kind), None)
            if r:
                nn_i = pool_idx[torch.cdist(r["final_latent"].flatten()[None], pool).argmin().item()]
                picks.append((r, nn_i))
    want = {(raw["episode"][k], raw["frame"][k]): None for r, nn_i in picks for k in (r["start"], r["goal"], nn_i)}
    eps, _ = load_meta(CACHE / REPO)
    for e, f, img in iter_frames(CACHE / REPO, eps[eps.episode_index.isin({e for e, _ in want})]):
        if (e, f) in want:
            want[(e, f)] = cv2.resize(img, (320, 240))

    def tile(k, text):
        img = want[(raw["episode"][k], raw["frame"][k])].copy()
        cv2.rectangle(img, (0, 0), (320, 22), (0, 0, 0), -1)
        cv2.putText(img, f"{text} pan {state[k, 0]:.0f}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return img

    grid = [np.hstack([tile(r["start"], "now"), tile(r["goal"], f"goal ({r['kind']})"),
                       tile(nn_i, f"imagined end, plan pan {r['pan_plan']:.0f} |")]) for r, nn_i in picks]
    out = OUT_DIR / f"plan_examples_{tag}_h{H}.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack(grid), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
