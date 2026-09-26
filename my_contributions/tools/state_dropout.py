"""Hide the robot's joint readings on a random half of training samples.

The proprioceptive state is a shortcut. Given the current joint angles, a large part of the
next 50 actions is predictable by continuing the trajectory the arm is already on — no need
to look at the pens, and certainly no need to read the instruction. That shortcut is always
available and always cheap, which is one reason plain behaviour cloning on this dataset
collapses to a single habitual motion.

Dropping the state on a fraction of samples removes the crutch on those samples: to predict
the actions the policy has to get the target location from the images, and which target from
the instruction. The remaining samples keep the state, so the policy still learns to use it
when it is there (it always is, at inference time).

The state is masked *after* normalization, so zero means "the dataset-average pose" — an
uninformative but in-distribution value — rather than an impossible joint configuration.

Enable via env var, then import before the trainer:

    STATE_DROPOUT_P=0.5 python -c \
      "import my_contributions.tools.state_dropout; \
       from lerobot.scripts.lerobot_train import main; main()" <args>
"""

import logging
import os

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE

DROPOUT_P = float(os.getenv("STATE_DROPOUT_P", "0.5"))
_DEBUG_STEPS = int(os.getenv("STATE_DROPOUT_DEBUG_STEPS", "3"))
_DEBUG_EVERY = int(os.getenv("STATE_DROPOUT_DEBUG_EVERY", "500"))
_calls = 0

_original_forward = SmolVLAPolicy.forward


def _state_dropout_forward(self, batch, *args, **kwargs):
    state = batch.get(OBS_STATE)
    if DROPOUT_P <= 0 or state is None or not self.training:
        return _original_forward(self, batch, *args, **kwargs)

    # One decision per sample, not per joint: the point is to remove the whole crutch on
    # some samples, not to add noise to every sample.
    keep = (torch.rand(state.shape[0], device=state.device) >= DROPOUT_P).float()
    shape = [state.shape[0]] + [1] * (state.dim() - 1)
    masked = dict(batch)
    masked[OBS_STATE] = state * keep.view(shape)

    global _calls
    _calls += 1
    if _calls <= _DEBUG_STEPS or _calls % _DEBUG_EVERY == 0:
        print(
            f"[state_dropout] call={_calls} p={DROPOUT_P} "
            f"hidden={1 - keep.mean().item():.2f} of batch",
            flush=True,
        )

    return _original_forward(self, masked, *args, **kwargs)


SmolVLAPolicy.forward = _state_dropout_forward
logging.info("state_dropout: enabled (p=%.3g)", DROPOUT_P)
