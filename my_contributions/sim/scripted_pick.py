"""Scripted expert that picks a named pen, plus the episode-layout sampler.

This is the piece that makes sim data worth collecting: the script *knows* which pen
is the target, so every episode is correctly language-grounded by construction. That
is exactly the supervision the 60 real episodes were too few to provide.
"""

from __future__ import annotations

import numpy as np
from pen_env import COLOURS, FPS, Layout, PenPickEnv
from sensor import apply_params, draw_params

# IK seed: the median real grasp posture. Seeding from here keeps the solver in the
# same elbow-up branch the real robot uses.
GRASP_SEED = np.array([0.0, 10.0, 28.6, 25.3, -83.3, 28.0])

# Gripper is in LeRobot percent. The fingertip pads reach a 10 mm gap (the pen's
# diameter) at 9.9%, so 9% closes just past the pen and holds it; commanding 12% or
# more never grips at all, and the measured success rate falls off a cliff there.
# NOTE: the real robot reports ~2% when closed on its pens, so the sim gripper channel
# is offset from the real one -- generate_dataset.py rescales it on the way out.
GRIPPER_OPEN = 28.0
GRIPPER_CLOSED = 9.0

# Pen placement region: the rectangle that solves at every approach tilt under
# pen_env.IK_LIMITS (see the sweep in tune_grasp.py). The +-19 cm of lateral span is
# ~38 cm of slot separation, against 18.5 cm between Ben's two taped real slots.
X_RANGE = (0.18, 0.28)
Y_RANGE = (-0.19, 0.19)
MIN_PEN_GAP = 0.075  # metres between pen centres, so the jaws can't straddle two
GRASP_DZ = 0.000  # vertical trim on the grasp waypoint, metres (swept; 0 is best)

# Joint-space offsets from the grasp pose, in degrees, read off the real episodes:
# hover-45 sits 9.3 deg above the grasp on shoulder_lift; lift+30 sits 21.1 deg above
# it and has rolled wrist_roll by +49 deg.
HOVER_DLIFT = (7.0, 13.0)
LIFT_DLIFT = (18.0, 26.0)
LIFT_DROLL = (38.0, 58.0)


def sample_layout(rng: np.random.Generator, n_pens: int = 3, target: str | None = None) -> Layout:
    """Sample pen positions with rejection sampling on the minimum separation."""
    colours = COLOURS[:n_pens]
    pts: list[np.ndarray] = []
    for _ in range(400):
        if len(pts) == n_pens:
            break
        p = np.array([rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)])
        if all(np.linalg.norm(p - q) >= MIN_PEN_GAP for q in pts):
            pts.append(p)
    while len(pts) < n_pens:  # fall back to a spread-out row
        pts.append(np.array([0.25, Y_RANGE[0] + len(pts) * 0.13]))

    rng.shuffle(pts)  # decorrelate colour from position
    pos, yaw = {}, {}
    for c, p in zip(colours, pts, strict=True):
        pos[c] = (float(p[0]), float(p[1]))
        # Pens lie roughly radially (as in the real scene) with +-18 deg of slop.
        yaw[c] = float(np.arctan2(p[1], p[0]) + np.deg2rad(rng.uniform(-18, 18)))
    return Layout(
        target=target if target is not None else colours[rng.integers(n_pens)],
        pen_pos=pos,
        pen_yaw=yaw,
        tilt_deg=float(rng.uniform(16.0, 32.0)),
    )


def _smoothstep(n: int) -> np.ndarray:
    """Minimum-jerk-ish 0->1 ramp, so joint trajectories look hand-teleoperated."""
    s = np.linspace(0.0, 1.0, n, endpoint=True)
    return s * s * (3.0 - 2.0 * s)


def plan(env: PenPickEnv, layout: Layout, rng: np.random.Generator, start_deg: np.ndarray):
    """Build the open-loop joint-space trajectory for one pick.

    The shape of the trajectory is copied from the real episodes rather than invented.
    Measuring the 60 real picks phase by phase shows the operator does NOT hover high
    above the pen with the tool held at the grasp orientation -- 1.5 s before the grasp
    the arm is already in the grasp posture, only ~9 deg higher on shoulder_lift, and
    1 s after the grasp it has raised shoulder_lift ~21 deg and rolled the wrist ~49 deg.

    So there is exactly ONE IK call here, for the grasp itself; the approach and the
    lift are joint-space offsets from it. That keeps every frame on the same manifold
    the real robot uses. Solving each waypoint independently instead put the arm in
    elbow-negative, wrist-folded postures that reach the same points by a route the
    real robot never takes.

    Returns (actions [T, 6], info), or (None, info) if IK fails.
    """
    pen_pos, _ = env.pen_pose(layout.target)
    yaw = layout.pen_yaw[layout.target]

    # Aim slightly off the pen's centre along its own axis, the way a person would.
    along = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    grasp_pt = pen_pos + along * rng.uniform(-0.015, 0.015)
    grasp_pt[2] = pen_pos[2] + rng.uniform(-0.001, 0.002) + GRASP_DZ

    seed = GRASP_SEED.copy()
    seed[0] = -np.degrees(np.arctan2(grasp_pt[1], grasp_pt[0]))
    q_grasp, err = env.ik_grasp(grasp_pt, yaw, layout.tilt_deg, seed)
    info = {"ik_err_mm": float(err * 1000)}
    if err > 0.004:
        return None, {**info, "fail": "ik"}

    q_grasp[5] = GRIPPER_OPEN
    q_hover = q_grasp.copy()
    q_hover[1] -= rng.uniform(*HOVER_DLIFT)  # raise the arm on shoulder_lift only

    q_close = q_grasp.copy()
    q_close[5] = GRIPPER_CLOSED

    q_lift = q_close.copy()
    q_lift[1] -= rng.uniform(*LIFT_DLIFT)
    q_lift[4] += rng.uniform(*LIFT_DROLL)  # the operator rolls the wrist while lifting

    segments = [
        (start_deg, q_hover, int(rng.integers(80, 105))),  # swing out from rest
        (q_hover, q_grasp, int(rng.integers(40, 55))),  # settle onto the pen
        (q_grasp, q_close, int(rng.integers(20, 30))),  # close the jaws
        (q_close, q_lift, int(rng.integers(40, 55))),  # lift and roll
    ]
    traj = []
    for a, b, n in segments:
        w = _smoothstep(n)[:, None]
        traj.append(a[None, :] * (1 - w) + b[None, :] * w)
    traj.append(np.repeat(q_lift[None, :], int(rng.integers(25, 45)), axis=0))  # hold
    actions = np.concatenate(traj).astype(np.float32)

    info["n_frames"] = len(actions)
    info["grasp_frame"] = segments[0][2] + segments[1][2]
    return actions, info


def rollout(env: PenPickEnv, layout: Layout, rng: np.random.Generator, start_deg: np.ndarray,
            cameras=("top", "wrist"), render: bool = True, sensor: bool = True):
    """Execute a planned pick, recording observations at `FPS`.

    Success = the target pen ends up lifted clear of the table and no other pen has
    been knocked more than a couple of centimetres.
    """
    env.reset(layout, rng, start_deg=start_deg)
    actions, info = plan(env, layout, rng, start_deg)
    if actions is None:
        return None, {**info, "fail": "ik"}

    pen_start = {c: env.pen_pose(c)[0].copy() for c in layout.pen_pos}
    states, frames = [], {c: [] for c in cameras}
    # Camera settings are drawn ONCE for the whole episode; only noise is per-frame.
    cam_p = {c: draw_params(c, rng) for c in cameras} if sensor else None
    for a in actions:
        states.append(env.state_deg())
        if render:
            for c in cameras:
                img = env.render(c)
                frames[c].append(apply_params(img, cam_p[c], rng) if sensor else img)
        env.step(a)

    target_z = float(env.pen_pose(layout.target)[0][2])
    lift = target_z - float(pen_start[layout.target][2])
    disturbed = max(
        (float(np.linalg.norm(env.pen_pose(c)[0][:2] - pen_start[c][:2]))
         for c in layout.pen_pos if c != layout.target),
        default=0.0,
    )
    info.update(
        target_lift_m=lift,
        distractor_move_m=disturbed,
        success=bool(lift > 0.04 and disturbed < 0.03),
        fps=FPS,
    )
    return {"action": actions, "state": np.array(states, np.float32), "frames": frames}, info
