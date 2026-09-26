"""MuJoCo digital twin of the SO-101 pen-selection scene.

The point of this env is to produce training episodes in the *same* action space as
the real dataset, so sim and real episodes can be co-trained in one LeRobotDataset:

  action = observation.state = [shoulder_pan, shoulder_lift, elbow_flex,
                                wrist_flex, wrist_roll, gripper]  in LeRobot degrees

The convention was validated against bklassen3434/pick_pen_v2_20260920_124400: the
so101_new_calib MJCF takes `deg2rad(action)` straight as qpos, and FK of the 60 real
grasp frames lands the tool site on the table at the two taped slots.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).parent
SCENE = HERE / "assets" / "pen_scene.xml"

ARM = slice(0, 5)  # the 5 arm joints; index 5 is the gripper
COLOURS = ["blue", "pink", "grey"]

FPS = 30
SUBSTEPS = 20  # 600 Hz physics / 30 Hz control

# The five arm joints are plain degrees in both conventions, but LeRobot's gripper
# channel is MotorNormMode.RANGE_0_100 (a percentage of the calibrated travel) while
# the MJCF joint is in radians over [-10, 100] degrees. Feeding percent straight in
# as degrees puts the jaws ~20 mm apart when the real robot reports "closed".
GRIP_DEG_LO, GRIP_DEG_HI = -10.0, 100.0

# Where a grasped pen ends up, expressed in the tool-site frame (x = approach,
# y = jaw hinge axis, z = jaw-closing direction). Both components matter:
#   x = +1.5 mm  the fingertip pads (add_jaw_pads.py) sit just past the site. Grasping
#                has to happen at the very tips; the jaws' concave throat only closes
#                to a pen's diameter ~17 mm further back, which would mean driving the
#                fingertips through the table.
#   z = +8.5 mm  the jaws are NOT symmetric about the site's axis. Only the moving jaw
#                swings, so the point where the two pads meet at a 10 mm gap sits 8.5 mm
#                off-axis. Aiming at the axis instead misses the pen by that much, which
#                is most of why the first few hundred scripted grasps closed on air.
GRASP_SITE_OFFSET = np.array([0.0015, 0.0, 0.0085])

# Joint box the IK is allowed to use. The MJCF's own limits are much wider, and
# unconstrained IK happily returns elbow-negative / wrist-folded postures that reach
# the same point by a route the real robot never takes -- which would put sim and real
# demonstrations on two different manifolds and defeat the point of co-training.
#
# elbow, wrist_flex and wrist_roll are pinned to the real robot's branch. shoulder_pan
# is deliberately left FREE: the real data only spans [-89.5, 12.7] because Ben's two
# taped slots were the only places a pen ever sat, and generating pens at new pan
# angles is the single most valuable thing this sim does. Constraining pan to the
# observed range shrank the usable workspace to a sliver and rejected 40% of layouts.
#          pan          lift         elbow       wrist_flex    wrist_roll
IK_LIMITS = np.deg2rad(np.array([
    [-110.0, 110.0], [-105.0, 35.0], [5.0, 97.0], [-12.0, 60.0], [-130.0, 20.0],
]))

# Median frame-0 pose of the real episodes, clipped to this model's joint limits.
# Units: 5 x degrees + gripper percent, i.e. LeRobot's own convention.
REST_DEG = np.array([-57.3, -100.0, 96.8, 21.1, 3.0, 14.7])


def to_rad(action: np.ndarray) -> np.ndarray:
    """LeRobot units ([5 x deg, gripper %]) -> MuJoCo qpos/ctrl radians."""
    a = np.asarray(action, float).copy()
    a[..., 5] = GRIP_DEG_LO + a[..., 5] * (GRIP_DEG_HI - GRIP_DEG_LO) / 100.0
    return np.deg2rad(a)


def from_rad(qpos: np.ndarray) -> np.ndarray:
    """MuJoCo qpos radians -> LeRobot units ([5 x deg, gripper %])."""
    a = np.rad2deg(np.asarray(qpos, float)).copy()
    a[..., 5] = (a[..., 5] - GRIP_DEG_LO) * 100.0 / (GRIP_DEG_HI - GRIP_DEG_LO)
    return a




def approach_frame(pen_pos: np.ndarray, pen_yaw: float, tilt_deg: float) -> np.ndarray:
    """Desired tool-site orientation for grasping a pen.

    The gripper comes down from above, tilted *outward* (away from the robot) by
    `tilt_deg`. That sign is not a guess: the real grasp frames have approach axes
    of [0.271, -0.043, -0.960] (slot A) and [0.397, 0.309, -0.834] (slot B), whose
    horizontal components point away from the base. The jaw hinge axis is held
    parallel to the pen so the jaws close across its width.

    Returns a 3x3 rotation matrix whose columns are the site's [x, y, z] axes:
    x = approach direction, y = hinge axis (must lie along the pen).
    """
    radial = np.array([np.cos(pen_yaw), np.sin(pen_yaw), 0.0])  # pen long axis, horizontal
    t = np.deg2rad(tilt_deg)
    # Approach: mostly straight down, tilted by `t` away from the robot (+radial).
    x_axis = np.sin(t) * radial - np.cos(t) * np.array([0.0, 0.0, 1.0])
    y_axis = radial - x_axis * (radial @ x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


@dataclass
class Layout:
    """One episode's scene configuration."""

    target: str  # which colour to pick
    pen_pos: dict = field(default_factory=dict)  # colour -> (x, y)
    pen_yaw: dict = field(default_factory=dict)  # colour -> radians
    tilt_deg: float = 25.0


class PenPickEnv:
    def __init__(self, n_pens: int = 3, width: int = 640, height: int = 480):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.n_pens = n_pens
        self.colours = COLOURS[:n_pens]
        self._renderer = mujoco.Renderer(self.model, height, width)
        self._site = self.model.site("gripperframe").id
        self._nominal_cam_pos = self.model.cam_pos.copy()
        self._nominal_cam_quat = self.model.cam_quat.copy()
        self._nominal_light_pos = self.model.light_pos.copy()
        self._nominal_light_dir = self.model.light_dir.copy()
        self._nominal_light_diffuse = self.model.light_diffuse.copy()
        self._nominal_mat_rgba = self.model.mat_rgba.copy()
        self._pen_qadr = {
            c: self.model.jnt_qposadr[self.model.joint(f"pen_{c}_free").id] for c in COLOURS
        }

    # ---------------------------------------------------------------- scene setup

    def set_pen(self, colour: str, x: float, y: float, yaw: float, z: float = 0.0035) -> None:
        a = self._pen_qadr[colour]
        self.data.qpos[a : a + 3] = [x, y, z]
        self.data.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qvel[a : a + 6] = 0.0

    def hide_pen(self, colour: str) -> None:
        """Park an unused pen far under the table, out of every camera's view."""
        self.set_pen(colour, 0.0, 0.0, 0.0, z=-5.0)

    def randomise_visuals(self, rng: np.random.Generator, strength: float = 1.0) -> None:
        """Domain randomisation: camera pose, lighting, table and pen shades.

        Nothing here changes the physics or the actions, only the pixels, so it is
        free diversity for sim2real.
        """
        m, s = self.model, strength
        m.cam_pos[:] = self._nominal_cam_pos
        m.cam_quat[:] = self._nominal_cam_quat
        top = m.camera("top").id
        m.cam_pos[top] += rng.normal(0, 0.03 * s, 3)
        # small random rotation of the top camera about all three axes
        ang = rng.normal(0, np.deg2rad(4.0) * s, 3)
        dq = np.empty(4)
        mujoco.mju_euler2Quat(dq, ang, "xyz")
        out = np.empty(4)
        mujoco.mju_mulQuat(out, self._nominal_cam_quat[top], dq)
        m.cam_quat[top] = out
        m.cam_fovy[top] = 52.0 + rng.normal(0, 2.0 * s)

        m.light_pos[:] = self._nominal_light_pos + rng.normal(0, 0.15 * s, self._nominal_light_pos.shape)
        m.light_dir[:] = self._nominal_light_dir
        # Per-light intensity jitter, NOT per-channel: the key is warm and the fill is
        # cool, so swinging their ratio by 25% swung the scene's colour temperature and
        # with it the rendered hue of a near-neutral pen. 12% keeps the variation in
        # brightness where it belongs.
        m.light_diffuse[:] = np.clip(
            self._nominal_light_diffuse * (1.0 + rng.normal(0, 0.12 * s, (m.nlight, 1))), 0.05, 1.0
        )

        m.mat_rgba[:] = self._nominal_mat_rgba
        for name in ("tablemat", "wallmat"):
            mid = m.material(name).id
            m.mat_rgba[mid, :3] = np.clip(
                self._nominal_mat_rgba[mid, :3] + rng.normal(0, 0.06 * s, 3), 0.05, 1.0
            )
        # Pen shades are jittered much more weakly than the table, and mostly in
        # BRIGHTNESS rather than hue. Colour is the entire task signal here: an
        # independent per-channel draw at 0.04 was enough to render the blue pen teal in
        # one episode and lavender in the next, which teaches an inconsistent
        # colour->word mapping. A shared luminance term plus a small per-channel
        # residual covers lighting variation without moving the hue off "blue".
        for c in COLOURS:
            mid = m.material(f"{c}_pen").id
            lum = rng.normal(0, 0.035 * s)
            m.mat_rgba[mid, :3] = np.clip(
                self._nominal_mat_rgba[mid, :3] * (1.0 + lum) + rng.normal(0, 0.006 * s, 3),
                0.0, 1.0,
            )

    def reset(self, layout: Layout, rng: np.random.Generator, start_deg: np.ndarray | None = None):
        mujoco.mj_resetData(self.model, self.data)
        q0 = REST_DEG.copy() if start_deg is None else np.asarray(start_deg, float).copy()
        self.data.qpos[:6] = to_rad(q0)
        self.data.ctrl[:] = to_rad(q0)
        for c in COLOURS:
            if c in layout.pen_pos:
                x, y = layout.pen_pos[c]
                self.set_pen(c, x, y, layout.pen_yaw[c])
            else:
                self.hide_pen(c)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(60):  # let the pens settle onto the table
            mujoco.mj_step(self.model, self.data)
        return self.state_deg()

    # ------------------------------------------------------------------ dynamics

    def state_deg(self) -> np.ndarray:
        """Joint positions in LeRobot units: 5 x degrees + gripper percent."""
        return from_rad(self.data.qpos[:6]).astype(np.float32)

    def step(self, action_deg: np.ndarray) -> np.ndarray:
        self.data.ctrl[:] = np.clip(
            to_rad(action_deg), self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1]
        )
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        return self.state_deg()

    def pen_pose(self, colour: str) -> tuple[np.ndarray, np.ndarray]:
        b = self.data.body(f"pen_{colour}")
        return b.xpos.copy(), b.xmat.reshape(3, 3).copy()

    def render(self, camera: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    # ------------------------------------------------------------------------ IK

    def ik(
        self,
        target_pos: np.ndarray,
        target_R: np.ndarray,
        q_init_deg: np.ndarray,
        iters: int = 200,
        rot_weight: float = 0.4,
        damping: float = 5e-3,
    ) -> tuple[np.ndarray, float]:
        """Damped-least-squares IK on the tool site over the 5 arm joints.

        The SO-101 is 5-DoF, so a general 6-D pose is unreachable. Orientation is
        down-weighted and the damping least-squares away whatever is left over,
        which in practice gives up a little wrist roll rather than position.

        Returns (joint angles in degrees, final position error in metres).
        """
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.data.qpos
        q = np.deg2rad(np.asarray(q_init_deg, float)[ARM]).copy()
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        lo, hi = IK_LIMITS[:, 0], IK_LIMITS[:, 1]
        qt, qc, err_rot = np.empty(4), np.empty(4), np.empty(3)

        for _ in range(iters):
            d.qpos[ARM] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos = d.site(self._site).xpos
            R = d.site(self._site).xmat.reshape(3, 3)
            e_pos = target_pos - pos
            mujoco.mju_mat2Quat(qt, np.ascontiguousarray(target_R.reshape(9)))
            mujoco.mju_mat2Quat(qc, np.ascontiguousarray(R.reshape(9)))
            mujoco.mju_negQuat(qc, qc)
            dq = np.empty(4)
            mujoco.mju_mulQuat(dq, qt, qc)
            mujoco.mju_quat2Vel(err_rot, dq, 1.0)

            err = np.concatenate([e_pos, rot_weight * err_rot])
            if np.linalg.norm(e_pos) < 3e-4 and np.linalg.norm(err_rot) < 0.03:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self._site)
            J = np.vstack([jacp[:, ARM], rot_weight * jacr[:, ARM]])
            step = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q = np.clip(q + np.clip(step, -0.25, 0.25), lo, hi)

        d.qpos[ARM] = q
        mujoco.mj_kinematics(m, d)
        final_err = float(np.linalg.norm(target_pos - d.site(self._site).xpos))
        out = np.asarray(q_init_deg, float).copy()
        out[ARM] = np.rad2deg(q)
        return out, final_err

    def ik_grasp(
        self, pinch_pos: np.ndarray, pen_yaw: float, tilt_deg: float, q_init_deg: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """IK for putting the *pinch point* (not the tool site) at `pinch_pos`."""
        R = approach_frame(pinch_pos, pen_yaw, tilt_deg)
        site_target = pinch_pos - R @ GRASP_SITE_OFFSET
        # Two phases: get the wrist into the right posture with orientation weighted
        # normally, then refine position with orientation nearly free. On a 5-DoF arm
        # the pose is over-constrained, and position is what decides whether the jaws
        # actually straddle the pen.
        q, _ = self.ik(site_target, R, q_init_deg, rot_weight=0.4)
        return self.ik(site_target, R, q, iters=120, rot_weight=0.06, damping=1e-4)
