"""Draw the green ring on the real top camera, inside lerobot-rollout.

The marker model was trained on top-camera frames with a ring burned in, so at inference
every top frame needs the same ring at the same spot. This patches SOFollower.get_observation
(the one place every camera frame passes through) to draw it, so the rest of lerobot-rollout
is untouched: rename map, recording, return-to-start, motor_retry all work as before. The
recorded episode stores the ringed frame, i.e. exactly what the policy saw.

The ring position comes from marker_aim.py, via the MARKER_UV env var ("u,v" in pixels of the
640x480 top frame). It is fixed for the whole attempt, exactly as in training.

Enable via an import shim, like motor_retry:
    MARKER_UV=287.0,382.1 .venv/bin/python -c \
      "import my_contributions.tools.marker_shim; from lerobot.scripts.lerobot_rollout import main; main()" <args>
"""

import logging
import os

from lerobot.robots.so_follower.so_follower import SOFollower
from my_contributions.tools.marker_overlay import draw_marker

TOP = "top"  # camera name in --robot.cameras

_raw = os.environ.get("MARKER_UV")
if not _raw:
    # Running the marker model without a ring would be the "no instruction" case it never saw.
    raise SystemExit("marker_shim: MARKER_UV is not set (run marker_aim.py first)")
UV = tuple(float(c) for c in _raw.split(","))

_original = SOFollower.get_observation


def _with_ring(self):
    obs = _original(self)
    if TOP not in obs:
        raise KeyError(f"marker_shim: no '{TOP}' camera in the observation ({list(obs)})")
    obs[TOP] = draw_marker(obs[TOP], UV)
    return obs


SOFollower.get_observation = _with_ring
logging.getLogger(__name__).warning("marker_shim: ring at %s on every '%s' frame", UV, TOP)
