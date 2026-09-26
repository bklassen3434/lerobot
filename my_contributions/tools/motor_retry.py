"""Retry a dropped Feetech status packet instead of killing the rollout.

Running SmolVLA on MPS stalls the process for ~0.4 s at every 50-action chunk boundary. The
Feetech serial bus does not tolerate that well: often enough, the next `sync_read` finds no
status packet waiting and raises

    ConnectionError: Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6]
    after 1 tries. [TxRxResult] There is no status packet!

which aborts the rollout mid-attempt. Sentry still saves the episode from its `finally`
block, so the data is not lost -- but the attempt is truncated, and a 5-second attempt may
not contain a whole reach-and-grasp.

`MotorsBus.sync_read` already takes a `num_retry` argument whose loop re-sends the packet.
Nothing in the robot layer passes it: `SOFollower.get_observation` calls
`self.bus.sync_read("Present_Position")` with the default of 0, and there is no config knob
to change that. So this patches the default rather than the call site, which keeps it out of
upstream code.

Enable via an import shim before the entrypoint, the same way `pi05_compat` and
`state_dropout` are wired in:

    .venv/bin/python -c \
      "import my_contributions.tools.motor_retry; \
       from lerobot.scripts.lerobot_rollout import main; main()" <args>

Set MOTOR_SYNC_READ_RETRIES=0 to disable without removing the import.
"""

import logging
import os

from lerobot.motors.motors_bus import MotorsBus

RETRIES = int(os.getenv("MOTOR_SYNC_READ_RETRIES", "3"))

_original_sync_read = MotorsBus.sync_read


def _retrying_sync_read(self, data_name, motors=None, *, normalize=True, num_retry=None):
    # Only supply a default; an explicit num_retry from a caller still wins.
    return _original_sync_read(
        self,
        data_name,
        motors,
        normalize=normalize,
        num_retry=RETRIES if num_retry is None else num_retry,
    )


MotorsBus.sync_read = _retrying_sync_read
logging.getLogger(__name__).info("[motor_retry] sync_read will retry %d times before raising", RETRIES)
print(f"[motor_retry] sync_read num_retry default = {RETRIES}", flush=True)
