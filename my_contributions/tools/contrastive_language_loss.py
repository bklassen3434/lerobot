"""Make ignoring the instruction expensive, by adding a counterfactual term to SmolVLA's loss.

Plain behaviour cloning only ever asks "did you predict the demonstrated actions?". Nothing
asks "did you predict them *because* of the instruction?". When two tasks share a scene and
differ by one word, ignoring the word and averaging the two targets is the cheaper solution,
and gradient descent takes it — which is exactly the failure measured across seven training
configs on this project (shoulder_pan moves 0.4-1.9 deg when the prompt changes, against the
43.4 deg needed to switch pens).

This patch adds a second forward pass per step with the language tokens *shuffled across the
batch*, and penalises the model when the demonstrated actions are still predicted well under
somebody else's instruction:

    loss = mean(L_correct) + lambda * mean( relu(margin_ratio * L_correct.detach() - L_wrong) )

The margin is relative rather than absolute: the wrong-instruction loss must be at least
`margin_ratio` times the correct-instruction loss, so the term auto-scales as training
progresses instead of needing a hand-tuned absolute value. Both passes share the same noise
and timestep, so the only difference between them is the instruction.

Rows whose shuffled instruction happens to be identical to the original (likely, with only
two tasks) are masked out — there is nothing to separate.

Cost: ~2x compute per step. Enable via env vars, then import before the trainer:

    CONTRASTIVE_LAMBDA=1.0 CONTRASTIVE_MARGIN=2.0 python -c \
      "import my_contributions.tools.contrastive_language_loss; \
       from lerobot.scripts.lerobot_train import main; main()" <args>
"""

import logging
import os

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

LAMBDA = float(os.getenv("CONTRASTIVE_LAMBDA", "1.0"))
MARGIN_RATIO = float(os.getenv("CONTRASTIVE_MARGIN", "2.0"))
# The trainer only logs its own metric set, so the extra loss terms would otherwise be
# invisible. Print the first few steps to stderr as proof the term is live and non-zero.
_DEBUG_STEPS = int(os.getenv("CONTRASTIVE_DEBUG_STEPS", "3"))
_DEBUG_EVERY = int(os.getenv("CONTRASTIVE_DEBUG_EVERY", "500"))
_calls = 0

_original_forward = SmolVLAPolicy.forward


def _contrastive_forward(self, batch, noise=None, time=None, reduction="mean"):
    # Only augment the training objective. Anything asking for per-sample losses (RA-BC
    # weighting) or supplying its own noise/time is left alone.
    if reduction != "mean" or LAMBDA <= 0 or noise is not None or time is not None:
        return _original_forward(self, batch, noise=noise, time=time, reduction=reduction)

    tokens = batch.get(OBS_LANGUAGE_TOKENS)
    if tokens is None or tokens.shape[0] < 2:
        return _original_forward(self, batch, reduction=reduction)

    # Share noise and timestep across both passes so the instruction is the only difference.
    actions = self.prepare_action(batch)
    noise = self.model.sample_noise(actions.shape, actions.device)
    time = self.model.sample_time(actions.shape[0], actions.device)

    per_sample, loss_dict = _original_forward(self, batch, noise=noise, time=time, reduction="none")

    # Counterfactual pass: every sample gets the next sample's instruction.
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    swapped = dict(batch)
    swapped[OBS_LANGUAGE_TOKENS] = torch.roll(tokens, shifts=1, dims=0)
    swapped[OBS_LANGUAGE_ATTENTION_MASK] = torch.roll(masks, shifts=1, dims=0)
    # `forward` mutates batch[ACTION] for the aloha variants; hand it a copy either way.
    swapped[ACTION] = batch[ACTION].clone()

    wrong_per_sample, _ = _original_forward(
        self, swapped, noise=noise, time=time, reduction="none"
    )

    # Rows where the shuffled instruction is genuinely different.
    differs = (swapped[OBS_LANGUAGE_TOKENS] != tokens).any(dim=1).float()

    penalty = torch.relu(MARGIN_RATIO * per_sample.detach() - wrong_per_sample) * differs
    penalty = penalty.sum() / differs.sum().clamp_min(1.0)

    loss = per_sample.mean() + LAMBDA * penalty

    loss_dict["loss"] = loss.item()
    loss_dict["bc_loss"] = per_sample.mean().item()
    loss_dict["wrong_instruction_loss"] = wrong_per_sample.mean().item()
    loss_dict["contrastive_penalty"] = penalty.item()
    loss_dict["frac_swapped"] = differs.mean().item()

    global _calls
    _calls += 1
    # The gap between `bc` and `wrong` widening over training is the signal that the policy
    # is starting to depend on the instruction, so keep printing periodically, not just once.
    if _calls <= _DEBUG_STEPS or _calls % _DEBUG_EVERY == 0:
        print(
            f"[contrastive] call={_calls} bc={per_sample.mean().item():.4f} "
            f"wrong={wrong_per_sample.mean().item():.4f} "
            f"penalty={penalty.item():.4f} frac_swapped={differs.mean().item():.2f} "
            f"total={loss.item():.4f}",
            flush=True,
        )
    return loss, loss_dict


SmolVLAPolicy.forward = _contrastive_forward
logging.info(
    "contrastive_language_loss: enabled (lambda=%.3g, margin_ratio=%.3g)", LAMBDA, MARGIN_RATIO
)
