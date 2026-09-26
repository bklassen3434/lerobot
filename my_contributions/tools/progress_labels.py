"""Turn a pick-the-pen dataset into per-frame progress labels in [0, 1].

The obvious label -- `frame_index / episode_length` -- is wrong here, and badly so. These
episodes are recorded on a fixed 15 s timer, and on `pick_pen_v2` the shape is always:

    frames   0-120   arm sits still at home                 (27% of every episode)
    frames 120-230   approach and align
    frame    ~230    gripper closes  <- the pen is picked
    frames 230-340   lift / hold
    frames 340-450   gripper opens, pen goes back, arm returns home

A clock label would call a frozen, untouched scene "28% done" and would score the *retreat*
higher than the grasp. So the labels here are anchored on events read straight out of the
action stream -- no hand annotation:

    t_move     first frame the arm actually moves      (summed |joint delta| > threshold)
    t_grasp    first gripper close after t_move        (the moment the pen is acquired)
    t_done     t_grasp + a short lift-confirm window   (pen is up: task complete)
    t_release  gripper reopens                         (end of the task proper)

    progress = 0                                    up to t_move
               0 -> 0.9  linearly                   t_move  -> t_grasp
               0.9 -> 1.0 linearly                  t_grasp -> t_done
               1.0                                  t_done  -> t_release
               (masked out)                         past t_done + HOLD_KEEP, or after t_release

Frames after the release are dropped rather than labelled. The arm returning home with an
empty gripper looks almost identical to the arm sitting at home at the start of the episode,
so labelling one 1.0 and the other 0.0 would just teach the model to average the two. Anything
downstream should therefore read the *peak* progress of a rollout, not its final value.

The two static stretches are capped for a different reason: class balance. The idle head is
27% of the dataset as one unchanging image labelled 0.0, and the post-lift hold is another
~110 frames of an unchanging image labelled 1.0 -- together 45% of the frames, all of them
trivial, drowning out the approach. IDLE_KEEP and HOLD_KEEP trim them to 30 and 45 frames,
which is why the dropped region on the right starts well before t_release rather than at it.

Usage:
    cd /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1 && \
      .venv/bin/python my_contributions/tools/progress_labels.py <repo_id> \
        [--plot outputs/progress_model/labels.png] [--save outputs/progress_model/labels.parquet]
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path.home() / ".cache/huggingface/lerobot"

MOVE_THRESHOLD = 0.5  # summed abs joint delta per frame (deg) that counts as "moving"
GRASP_LEVEL = 0.5  # fraction of the way from closed to open that counts as "closed"
REACH_THRESHOLD = 250.0  # summed joint travel (deg) from the start pose before a close counts
LIFT_FRAMES = 15  # 0.5 s after the grasp before we call the pick complete
IDLE_KEEP = 30  # frames of "nothing happening yet" kept before t_move
HOLD_KEEP = 45  # frames of "pen is up, task done" kept after t_done
GRASP_VALUE = 0.9  # progress credited for the grasp itself; the last 0.1 is the lift
ROLLOUT_PARTIAL = 0.4  # credit for "reached the pen and closed, but never lifted it"


def _load(repo_id: str) -> tuple[pd.DataFrame, dict[int, str]]:
    root = Path(repo_id) if Path(repo_id).is_dir() else CACHE / repo_id
    if not root.is_dir():
        raise SystemExit(f"dataset not found: {root}")
    files = sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files]).reset_index(drop=True)
    tasks = pd.read_parquet(root / "meta/tasks.parquet").reset_index()
    return df, {int(r.task_index): r.task for r in tasks.itertuples()}


def events(repo_id: str) -> pd.DataFrame:
    """Locate t_move / t_grasp / t_done / t_release in every episode."""
    df, task_names = _load(repo_id)

    rows = []
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index").reset_index(drop=True)
        actions = np.stack(g["action"].values)[:, :6]
        grip = actions[:, 5]
        n = len(g)

        speed = np.abs(np.diff(actions, axis=0)).sum(axis=1)
        moving = np.where(speed > MOVE_THRESHOLD)[0]
        t_move = int(moving[0]) if len(moving) else 0

        # Per-episode threshold: the gripper's open/closed values drift between sessions, so a
        # fixed cutoff mislabels whole blocks. Percentiles ignore the transition frames.
        lo, hi = np.percentile(grip, 10), np.percentile(grip, 90)
        shut = grip < lo + GRASP_LEVEL * (hi - lo)

        # The grasp is an open->closed *transition*, not just "closed". In 5 of 60 episodes the
        # operator left the gripper shut at home and opened it on the way to the pen; taking the
        # first closed frame there puts the grasp at the home pose, ~55 deg from any pen.
        # ...and it has to happen with the arm reached out, not parked. A demonstrator never
        # closes the gripper at the rest pose, but a failing policy does it constantly: in the
        # first rollout, 5 of 13 "grasps" were the gripper shutting at home (lift ~ -100,
        # elbow ~ 95), nowhere near a pen. REACH_THRESHOLD is set from the training data, where
        # the summed joint travel at the moment of a real grasp is 305-386 (n=60); the bogus
        # home closures sat at 17-71. This is a heuristic cross-check, not ground truth --
        # use --grasp-frames to confirm by eye on any rollout before trusting it.
        travel = np.abs(actions - actions[0]).sum(axis=1)
        reached = travel > REACH_THRESHOLD

        after_open = np.where(~shut[t_move:])[0]
        t_grasp = -1
        if len(after_open):
            opened = t_move + int(after_open[0])
            closing = np.where(shut[opened:] & reached[opened:])[0]
            if len(closing):
                t_grasp = opened + int(closing[0])

        t_release = n - 1
        if t_grasp > 0:
            reopen = np.where(~shut[t_grasp:])[0]
            if len(reopen):
                t_release = int(t_grasp + reopen[0])

        rows.append(
            {
                "ep": int(ep),
                "n": n,
                "task": task_names[int(g["task_index"].iloc[0])],
                "t_move": t_move,
                "t_grasp": t_grasp,
                "t_done": min(t_grasp + LIFT_FRAMES, t_release) if t_grasp > 0 else -1,
                "t_release": t_release,
                "pan_at_grasp": float(actions[t_grasp, 0]) if t_grasp > 0 else np.nan,
                "grip_open": float(hi),
                "grip_closed": float(lo),
            }
        )
    return pd.DataFrame(rows)


def rollout_labels(repo_id: str, outcomes: dict, tasks: list[str]) -> pd.DataFrame:
    """Per-(frame, colour) targets for a ROLLOUT, from hand-checked outcomes.

    Demonstrations only ever show success, which is why the first progress model scored "the
    gripper closed next to the pink pen" as 0.92 even though the pen never left the table.
    Rollouts supply the missing half: attempts that reach, close, and fail.

    Long format -- one row per (frame, colour) -- because a rollout frame carries a different
    target for each colour, and the informative one is usually the *negative*:

        colour the gripper did NOT close on   -> 0.0 for the whole episode (hard negative)
        colour it did close on, after the lift window
              ...and the pen came up          -> 1.0   (a real success, same as a demo)
              ...and the pen stayed put       -> ROLLOUT_PARTIAL
        colour it did close on, before that   -> no row; the approach is genuinely ambiguous
                                                 and demo data already teaches that ramp

    The frames between the grasp and t_done are left out on purpose: at the instant the
    gripper shuts, a success and a failure look identical. The difference only becomes visible
    once the pen should have moved and hasn't.
    """
    ev = events(repo_id).set_index("ep")
    rows = []
    for ep_str, o in outcomes.items():
        ep = int(ep_str)
        if ep not in ev.index:
            continue
        n, t_grasp = int(ev.loc[ep].n), int(ev.loc[ep].t_grasp)
        grabbed = o.get("grabbed")
        t = np.arange(n)

        if grabbed is None:
            # Never closed on a pen: no progress toward either colour, all episode.
            for c in tasks:
                rows.append(pd.DataFrame({"ep": ep, "frame": t, "task": c, "target": 0.0}))
            continue

        done = min(t_grasp + LIFT_FRAMES, n - 1)
        lifted = o.get("lifted")
        for c in tasks:
            if c != grabbed:
                rows.append(pd.DataFrame({"ep": ep, "frame": t, "task": c, "target": 0.0}))
            elif lifted is None:
                # Outcome genuinely unclear from the video -- the pen is in the jaws but never
                # visibly leaves the table. Don't invent a label; the hard negative on the
                # other colour is still valid and is kept.
                continue
            else:
                after = t[t >= done]
                rows.append(
                    pd.DataFrame({"ep": ep, "frame": after, "task": c, "target": 1.0 if lifted else ROLLOUT_PARTIAL})
                )
    return pd.concat(rows).reset_index(drop=True)


def label_episode(
    e: pd.Series, idle_keep: int = IDLE_KEEP, hold_keep: int = HOLD_KEEP
) -> tuple[np.ndarray, np.ndarray]:
    """(progress, keep) arrays of length e.n for one episode."""
    t = np.arange(e.n)
    p = np.zeros(e.n, dtype=np.float32)

    approach = (t > e.t_move) & (t <= e.t_grasp)
    p[approach] = GRASP_VALUE * (t[approach] - e.t_move) / max(1, e.t_grasp - e.t_move)

    lift = (t > e.t_grasp) & (t <= e.t_done)
    p[lift] = GRASP_VALUE + (1 - GRASP_VALUE) * (t[lift] - e.t_grasp) / max(1, e.t_done - e.t_grasp)

    p[t > e.t_done] = 1.0

    # The idle head and the post-lift hold are both long and both visually static, so without
    # capping them the model spends most of its capacity on "nothing is happening" (label 0)
    # and "still holding" (label 1) and barely sees the approach, which is the part that
    # carries the signal.
    keep = (t >= e.t_move - idle_keep) & (t <= min(e.t_release, e.t_done + hold_keep))
    return p, keep


def frame_labels(
    repo_id: str, ev: pd.DataFrame, idle_keep: int = IDLE_KEEP, hold_keep: int = HOLD_KEEP
) -> pd.DataFrame:
    """One row per kept frame: episode, frame index within the episode, task, progress."""
    out = []
    for e in ev.itertuples():
        if e.t_grasp < 0:
            continue
        p, keep = label_episode(pd.Series(e._asdict()), idle_keep, hold_keep)
        idx = np.where(keep)[0]
        out.append(
            pd.DataFrame(
                {
                    "ep": e.ep,
                    "frame": idx.astype(np.int32),
                    "task": e.task,
                    "progress": p[idx],
                }
            )
        )
    return pd.concat(out).reset_index(drop=True)


def report(ev: pd.DataFrame, labels: pd.DataFrame) -> None:
    bad = ev[ev.t_grasp < 0]
    print(f"episodes {len(ev)} | frames {ev.n.sum()} | tasks {sorted(ev.task.unique())}")
    print(
        f"  t_move    median {ev.t_move.median():.0f}  "
        f"[{ev.t_move.min()}-{ev.t_move.max()}]"
    )
    ok = ev[ev.t_grasp >= 0]
    print(f"  t_grasp   median {ok.t_grasp.median():.0f}  [{ok.t_grasp.min()}-{ok.t_grasp.max()}]")
    print(
        f"  t_release median {ok.t_release.median():.0f}  "
        f"[{ok.t_release.min()}-{ok.t_release.max()}]"
    )
    print(f"  approach  median {(ok.t_grasp - ok.t_move).median():.0f} frames")
    if len(bad):
        print(f"\n  {len(bad)} episode(s) with NO detectable grasp -- dropped: {list(bad.ep)}")

    kept, total = len(labels), int(ev.n.sum())
    print(f"\nlabelled frames {kept} / {total} ({100 * kept / total:.0f}%)")
    hist = np.histogram(labels.progress, bins=10, range=(0, 1))[0]
    print("  progress histogram (0.0 -> 1.0):", " ".join(f"{h:5d}" for h in hist))
    for task, g in labels.groupby("task"):
        print(f"  {task:>6s}: {len(g):6d} frames, {g.ep.nunique()} episodes")

    # Sanity: pan at the grasp should cluster by slot, not smear. A smear means the grasp
    # detector is firing on something other than the pen pickup.
    print("\ngrasp pan by task (should show both slots for every colour):")
    for task, g in ok.groupby("task"):
        v = np.sort(g.pan_at_grasp.values)
        print(f"  {task:>6s}: {np.round(v, 1)}")


def plot(ev: pd.DataFrame, repo_id: str, out: Path, n: int = 6) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, _ = _load(repo_id)
    ok = ev[ev.t_grasp >= 0]
    picks = ok.ep.iloc[np.linspace(0, len(ok) - 1, n).astype(int)]

    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.1 * len(picks)), sharex=True)
    for ax, ep in zip(np.atleast_1d(axes), picks, strict=False):
        e = ev[ev.ep == ep].iloc[0]
        g = df[df.episode_index == ep].sort_values("frame_index")
        a = np.stack(g["action"].values)
        p, keep = label_episode(e)

        ax.plot(p, lw=2, color="tab:blue", label="progress")
        ax.plot(a[:, 5] / max(1e-6, e.grip_open), lw=1, color="tab:grey", label="gripper (norm)")
        ax.plot((a[:, 0] - a[:, 0].min()) / np.ptp(a[:, 0]), lw=1, color="tab:orange", alpha=0.6, label="pan (norm)")
        ax.fill_between(np.arange(e.n), 0, 1, where=~keep, color="red", alpha=0.08)
        for t, c in [(e.t_move, "k"), (e.t_grasp, "g"), (e.t_release, "r")]:
            ax.axvline(t, color=c, ls=":", lw=1)
        ax.set_ylabel(f"ep {ep}\n{e.task}", fontsize=8)
        ax.set_ylim(-0.1, 1.2)
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=3, loc="upper left")
    np.atleast_1d(axes)[-1].set_xlabel("frame  (dotted: t_move / t_grasp / t_release; red = dropped)")
    fig.suptitle(f"progress labels -- {repo_id}", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


def grasp_frames(repo_id: str, ev: pd.DataFrame, out: Path) -> None:
    """Contact sheet of the top-camera frame at each detected grasp.

    The action-derived grasp is a heuristic, and on rollouts it is wrong often enough to
    matter. This is the check that actually settles which pen the gripper closed on -- run it
    before believing any `grasped` column.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=str(CACHE / repo_id))
    starts = [int(s) for s in ds.meta.episodes["dataset_from_index"]]
    ok = ev[ev.t_grasp >= 0]
    if not len(ok):
        print("no grasps detected -- nothing to render")
        return

    cols = min(4, len(ok))
    rows = (len(ok) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows))
    for ax, e in zip(np.atleast_1d(axes).ravel(), ok.itertuples(), strict=False):
        img = ds[starts[e.ep] + int(e.t_grasp)]["observation.images.top"]
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.set_title(f"ep {e.ep} [{e.task}]  frame {e.t_grasp}  pan {e.pan_at_grasp:.0f}", fontsize=8)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    fig.suptitle(f"frame at each detected grasp -- {repo_id}", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("--idle-keep", type=int, default=IDLE_KEEP)
    p.add_argument("--hold-keep", type=int, default=HOLD_KEEP)
    p.add_argument("--plot", default=None, help="write a diagnostic figure here")
    p.add_argument("--save", default=None, help="write per-frame labels to this parquet")
    p.add_argument("--grasp-frames", default=None, help="contact sheet of the frame at each grasp")
    a = p.parse_args()

    ev = events(a.repo_id)
    labels = frame_labels(a.repo_id, ev, a.idle_keep, a.hold_keep)
    report(ev, labels)

    if a.save:
        out = Path(a.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(out)
        print(f"\nwrote {out}")
    if a.plot:
        plot(ev, a.repo_id, Path(a.plot))
    if a.grasp_frames:
        grasp_frames(a.repo_id, ev, Path(a.grasp_frames))


if __name__ == "__main__":
    main()
