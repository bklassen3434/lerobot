"""Add fingertip pad collision geoms to the SO-101 jaws.

Why this is needed: MuJoCo collides meshes by their *convex hull*, which fills in the
concave inner face of each jaw. The hull faces are 16-20 mm apart even with the gripper
fully closed, and the only place they come within a pen's diameter is ~17 mm behind the
fingertips -- which would require driving the tips through the table to reach a pen lying
on it. mujoco_menagerie solves this for the SO-ARM100 by adding explicit box pads; this
script does the same for the so101_new_calib model, sizing and placing them from the
measured mesh geometry rather than by eye.

Pads are group 3 (collision), so they do not appear in any rendered camera view.

Idempotent: re-running replaces the previously inserted block.
"""

from __future__ import annotations

import pathlib
import re

import mujoco
import numpy as np

ARM_XML = pathlib.Path(__file__).parent / "assets" / "so101_arm.xml"
MARK_OPEN = "<!-- BEGIN generated jaw pads (add_jaw_pads.py) -->"
MARK_CLOSE = "<!-- END generated jaw pads -->"

# Pad geometry, expressed in the tool-site frame (x = approach/out, y = hinge axis,
# z = jaw-closing direction), measured with the gripper at 0 deg:
#   fixed-jaw hull inner face   z ~ -0.0004   at x = -0.002
#   moving-jaw hull inner face  z ~ +0.0162   at x = -0.002
# A 3.5 mm pad on each face leaves a 9.6 mm gap at 0 deg, so a 10 mm pen is gripped
# just past closure, at the fingertips where the jaws can actually reach the table.
# The pads must sit where a pen rests when the fingertips are touching the table.
# Tips are at site-x = +0.007; a pen's centre is one radius (5 mm) above the table, so
# along an approach tilted ~25 deg from vertical the pen is 5/cos(25) = 5.5 mm back
# from the tips, i.e. at site-x = +0.0015. Placing the pads anywhere further back makes
# the grasp geometrically impossible: the tips bottom out before the pen reaches them.
PAD_X = 0.0015
PAD_HALF = np.array([0.006, 0.008, 0.00175])  # half-extents along site x, y, z
PAD_T = 2 * PAD_HALF[2]


def _pad_xml(name: str, body_id: int, model, data, site_id: int, z_centre: float) -> str:
    """Express a site-frame box in `body_id`'s local frame and emit the geom XML."""
    Rs = data.site_xmat[site_id].reshape(3, 3)
    ps = data.site_xpos[site_id]
    Rb = data.xmat[body_id].reshape(3, 3)
    pb = data.xpos[body_id]

    centre_site = np.array([PAD_X, 0.0, z_centre])
    pos = Rb.T @ (Rs @ centre_site + ps - pb)
    Rrel = Rb.T @ Rs
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(Rrel.reshape(9)))
    return (
        f'<geom name="{name}" type="box" group="3" '
        f'size="{PAD_HALF[0]:.5f} {PAD_HALF[1]:.5f} {PAD_HALF[2]:.5f}" '
        f'pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" '
        f'quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}" '
        f'friction="1.6 0.05 0.002" condim="4" solimp="0.97 0.995 0.001" solref="0.004 1" '
        f'rgba="0.2 0.8 0.2 1"/>'
    )


def main() -> None:
    src = ARM_XML.read_text()
    src = re.sub(f"{re.escape(MARK_OPEN)}.*?{re.escape(MARK_CLOSE)}\n?", "", src, flags=re.S)

    ARM_XML.write_text(src)  # mesh paths are relative, so the model must load from disk
    model = mujoco.MjModel.from_xml_path(str(ARM_XML))
    data = mujoco.MjData(model)
    data.qpos[:6] = np.deg2rad([8.4, 7.0, 35.7, 31.0, -84.9, 0.0])  # gripper at 0 deg
    mujoco.mj_forward(model, data)

    site = model.site("gripperframe").id
    Rs, ps = data.site_xmat[site].reshape(3, 3), data.site_xpos[site]

    # Re-measure the two hull faces at PAD_X rather than trusting the numbers above.
    def face(geom_id: int, take_max: bool) -> float:
        mid = model.geom_dataid[geom_id]
        V = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        V = V.reshape(-1, 3)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[geom_id])
        b = model.geom_bodyid[geom_id]
        W = (data.xmat[b].reshape(3, 3) @ (R.reshape(3, 3) @ V.T + model.geom_pos[geom_id][:, None])).T
        W = W + data.xpos[b]
        L = (W - ps) @ Rs
        sel = L[(np.abs(L[:, 0] - PAD_X) < 0.004) & (np.abs(L[:, 1]) < PAD_HALF[1])]
        return float(sel[:, 2].max() if take_max else sel[:, 2].min())

    def collision_geom(body_name: str, mesh_name: str) -> int:
        """Geom ids shift when this file is included into a scene, so look them up."""
        bid = model.body(body_name).id
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid or model.geom_group[g] != 3:
                continue
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[g])) == mesh_name:
                return g
        raise LookupError(f"no collision geom for {body_name}/{mesh_name}")

    fixed_face = face(collision_geom("gripper", "wrist_roll_follower_so101_v1"), take_max=True)
    moving_face = face(collision_geom("moving_jaw_so101_v1", "moving_jaw_so101_v1"), take_max=False)
    print(f"measured at site-x={PAD_X * 1000:+.1f}mm: fixed face z={fixed_face * 1000:+.2f}mm, "
          f"moving face z={moving_face * 1000:+.2f}mm, hull gap={(moving_face - fixed_face) * 1000:.2f}mm")

    z_fixed = fixed_face + PAD_HALF[2]
    z_moving = moving_face - PAD_HALF[2]
    print(f"pad gap at gripper=0deg: {(z_moving - z_fixed - PAD_T) * 1000:.2f}mm  (pen is 10.0mm)")

    pads = {
        "gripper": _pad_xml("fixed_jaw_pad", model.body("gripper").id, model, data, site, z_fixed),
        "moving_jaw_so101_v1": _pad_xml(
            "moving_jaw_pad", model.body("moving_jaw_so101_v1").id, model, data, site, z_moving
        ),
    }

    out = src
    for body, geom in pads.items():
        # Insert just after the body's opening tag's first child line (the joint), so
        # the geom inherits the body frame.
        m = re.search(rf'(<body name="{body}"[^>]*>\n)', out)
        assert m, f"body {body} not found"
        block = f"{MARK_OPEN}\n                  {geom}\n                  {MARK_CLOSE}\n"
        out = out[: m.end()] + block + out[m.end() :]

    ARM_XML.write_text(out)
    mujoco.MjModel.from_xml_path(str(ARM_XML))  # fails loudly if the XML is malformed
    print(f"wrote {ARM_XML}")
    print(f"PINCH_OFFSET should be {-PAD_X:.4f}")


if __name__ == "__main__":
    main()
