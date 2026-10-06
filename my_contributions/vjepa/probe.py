"""Step 1b of the V-JEPA 2 world-model experiment: can simple linear read-outs recover what matters?

Reads outputs/embeddings.npz (from embed.py) and fits one ridge regression ("linear probe") per
question, on 3 kinds of features:
  - pixels: the frame shrunk to 32x24 RGB  (dumb baseline; V-JEPA 2 has to beat this)
  - mean:   V-JEPA 2 tokens averaged over the whole image
  - grid:   V-JEPA 2 tokens averaged over a 4x4 grid (keeps rough position)

Questions:
  joints       the 6 joint angles                                   (R^2, 1 = perfect, 0 = useless)
  progress     how far through the episode we are                   (R^2)
  pen_layout   is the blue pen on the left?                         (accuracy, chance ~50%)
  target_side  which side is the arm going to? per quarter of the episode (accuracy)
               Early on the arm is idle, so ~chance is the CORRECT answer there; anything
               much higher means the probe is cheating off something other than the arm.

Train/test is split by EPISODE (every 4th episode of each task x layout group is held out), so
the probe can't memorise neighbouring frames of the same episode.

    cd /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1 && \
        .venv/bin/python my_contributions/vjepa/probe.py
"""

import json
import pathlib

import numpy as np
import torch

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
FEATURES = ["pixels", "mean", "grid"]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ALPHAS = np.logspace(-4, 2, 13)  # ridge strength, as a multiple of feature dimension


def split_episodes(episode: np.ndarray, group: np.ndarray, every: int = 4) -> np.ndarray:
    """Boolean test mask: every `every`-th episode within each group goes to test."""
    test_eps = set()
    for g in np.unique(group):
        eps = np.unique(episode[group == g])
        test_eps |= set(eps[every - 1 :: every].tolist())
    return np.isin(episode, list(test_eps))


class Ridge:
    """Kernel-form ridge (fast when features >> samples). One eigendecomposition, any lambda."""

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "Ridge":
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        self.x = (x - self.mu) / self.sd
        self.y_mu = y.mean(0)
        k = (self.x @ self.x.T).double()
        self.evals, self.evecs = torch.linalg.eigh(k)
        self.uty = self.evecs.T @ (y - self.y_mu).double()
        return self

    def predict(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        lam = alpha * self.x.shape[1]
        coef = self.evecs @ (self.uty / (self.evals + lam)[:, None])
        k = (((x - self.mu) / self.sd) @ self.x.T).double()
        return (k @ coef).float() + self.y_mu


def r2(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def acc(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((pred > 0) == (y > 0)).float().mean(0)


def probe(x, y, train, test, val_inner, score):
    """Pick ridge strength on an inner split of the training episodes, refit, score on test."""
    inner = Ridge().fit(x[train & ~val_inner], y[train & ~val_inner])
    best = max(ALPHAS, key=lambda a: score(inner.predict(x[train & val_inner], a), y[train & val_inner]).mean())
    model = Ridge().fit(x[train], y[train])
    return model.predict(x[test], best), best


def main() -> None:
    d = np.load(OUT_DIR / "embeddings.npz")
    ok = d["blue_left"] >= 0
    if (~ok).any():
        print(f"dropping {len(np.unique(d['episode'][~ok]))} episodes where the pen detector failed")
    d = {k: v[ok] for k, v in d.items()}

    episode, task, blue_left, progress = d["episode"], d["task"], d["blue_left"], d["progress"]
    group = np.char.add(task.astype(str), blue_left.astype(str))
    test = split_episodes(episode, group)
    train = ~test
    # inner validation (for picking ridge strength) = every 4th TRAINING episode per group
    val_inner = np.zeros_like(test)
    val_inner[train] = split_episodes(episode[train], group[train])

    target_left = (task == "blue") == (blue_left == 1)
    print(f"{len(np.unique(episode))} episodes ({len(np.unique(episode[test]))} held out), {len(episode)} frames")
    print("episodes per task x blue-on-left:", {g: len(np.unique(episode[group == g])) for g in np.unique(group)})

    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))  # noqa: E731
    y_joints = t(d["state"])
    y_prog = t(progress)[:, None]
    y_layout = t(np.where(blue_left == 1, 1.0, -1.0))[:, None]
    y_target = t(np.where(target_left, 1.0, -1.0))[:, None]
    quarter = np.minimum((progress * 4).astype(int), 3)[test]

    results: dict = {}
    for name in FEATURES:
        x = t(d[name])
        res: dict = {"dim": x.shape[1]}
        pred, a = probe(x, y_joints, train, test, val_inner, r2)
        res["joints_r2"] = dict(zip(JOINTS, r2(pred, y_joints[test]).tolist()))
        pred, _ = probe(x, y_prog, train, test, val_inner, r2)
        res["progress_r2"] = r2(pred, y_prog[test]).item()
        pred, _ = probe(x, y_layout, train, test, val_inner, acc)
        res["pen_layout_acc"] = acc(pred, y_layout[test]).item()
        pred, _ = probe(x, y_target, train, test, val_inner, acc)
        hit = ((pred[:, 0] > 0) == (y_target[test, 0] > 0)).numpy()
        res["target_side_acc_by_quarter"] = [float(hit[quarter == q].mean()) for q in range(4)]
        results[name] = res
        print(f"  done: {name}")

    chance_layout = max(y_layout[test].gt(0).float().mean().item(), y_layout[test].lt(0).float().mean().item())
    chance_target = max(y_target[test].gt(0).float().mean().item(), y_target[test].lt(0).float().mean().item())
    results["chance"] = {"pen_layout": chance_layout, "target_side": chance_target}

    print("\n                       " + "".join(f"{n:>10}" for n in FEATURES))
    for j in JOINTS:
        print(f"R2 {j:<19}" + "".join(f"{results[n]['joints_r2'][j]:>10.3f}" for n in FEATURES))
    print(f"R2 {'progress':<19}" + "".join(f"{results[n]['progress_r2']:>10.3f}" for n in FEATURES))
    print(
        f"acc pen layout        " + "".join(f"{results[n]['pen_layout_acc']:>10.1%}" for n in FEATURES)
        + f"   (chance {chance_layout:.0%})"
    )
    for q in range(4):
        print(
            f"acc target, Q{q + 1}        " + "".join(f"{results[n]['target_side_acc_by_quarter'][q]:>10.1%}" for n in FEATURES)
            + (f"   (chance {chance_target:.0%})" if q == 0 else "")
        )

    (OUT_DIR / "probe_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_DIR / 'probe_results.json'}")


if __name__ == "__main__":
    main()
