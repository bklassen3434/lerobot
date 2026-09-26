"""Measure how much a trained SmolVLA checkpoint's output depends on the instruction.

Runs the policy on real frames from the training dataset, once per candidate instruction,
with identical observations and identical flow-matching noise. If the predicted action
chunks are the same no matter which colour you ask for, the language pathway is dead — and
you can see that in five minutes on a laptop instead of an afternoon of robot trials.

The comparison number that matters is `Δ(prompt) / action std`: the mean absolute difference
between two prompts' action chunks, expressed as a fraction of how much the actions vary
across the dataset. ~0 means the instruction is ignored; approaching 1 means it drives the
behaviour as much as everything else combined.

Usage:
    cd /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1 && \
      .venv/bin/python my_contributions/tools/language_sensitivity_probe.py \
        outputs_from_modal/smolvla_pick_pen_20000 [n_frames]
"""

import sys
from pathlib import Path

import numpy as np
import torch

import my_contributions.tools.pi05_compat  # noqa: F401  (registers pi05_base's step aliases)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats

REPO_ID = "bklassen3434/pick_pen_v2_20260920_124400"  # override with a 3rd CLI arg
CACHE = "/Users/benklassen/.cache/huggingface/lerobot"
RENAME_MAP = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}
# The real instructions the dataset was recorded with. `COLOURS` is how many of these are
# compared against each other; anything after that is a control the model never saw.
# Must match the task strings the checkpoint was TRAINED on, or you are probing strings the
# model never saw. v2 was relabelled to bare colour words on 2026-09-21.
PROMPTS = [
    "blue",
    "pink",
    "banana",  # nonsense control
]
COLOURS = 2

# In the v2 dataset the two taped slots sit 43.4 deg apart in shoulder_pan, so that is the
# distance the instruction has to move the arm to switch which pen gets picked. Reporting
# Δ against it turns an abstract score into "how much of the way there did it get".
SLOT_SEPARATION_DEG = 43.4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main(ckpt: str, n_frames: int = 12, repo_id: str = REPO_ID) -> None:
    ds = LeRobotDataset(repo_id, root=f"{CACHE}/{repo_id}")

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    policy_cls = get_policy_class(cfg.type)

    # LoRA checkpoints hold only adapter weights, so load the base policy the adapter was
    # trained on and wrap it, the same way lerobot-rollout does.
    if (Path(ckpt) / "adapter_config.json").is_file():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(ckpt)
        policy = policy_cls.from_pretrained(peft_config.base_model_name_or_path, config=cfg)
        policy = PeftModel.from_pretrained(policy, ckpt, config=peft_config)
    else:
        policy = policy_cls.from_pretrained(ckpt, config=cfg)
    policy = policy.to(DEVICE).eval()

    # pi05 is trained via --policy.type, so its config already carries the dataset's own
    # camera keys and must NOT be renamed. smolvla_base bakes in camera1/2/3 and must be.
    expected = [k for k in cfg.input_features if "image" in k]
    if any("images.top" in k for k in expected):
        rename_map = {}  # config already uses our dataset's own keys (pi05 via --policy.type)
    else:
        # Map our two views onto whatever the checkpoint expects, in feature order. Covers
        # smolvla_base (camera1/2/3) and pi05_base (base_0_rgb, left_wrist_0_rgb, ...).
        ours = ["observation.images.top", "observation.images.wrist"]
        rename_map = dict(zip(ours, expected))

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=ckpt,
        dataset_stats=rename_stats(ds.meta.stats, rename_map),
        preprocessor_overrides={
            "device_processor": {"device": DEVICE},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    # Sample the first frame of episodes spread across the dataset, so we cover all three
    # colours and all three pen positions.
    starts = [int(ds.meta.episodes["dataset_from_index"][i]) for i in range(ds.meta.total_episodes)]
    picks = np.linspace(0, len(starts) - 1, n_frames).astype(int)

    # Fixed noise: SmolVLA denoises from a random sample, so without this the run-to-run
    # jitter would swamp the prompt effect we're trying to measure.
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=gen).to(DEVICE)

    # Only the first 6 action dims are real joints — these policies pad to max_action_dim=32
    # and the padding never moves, so averaging over all 32 would dilute the signal ~5x.
    # predict_action_chunk returns NORMALIZED actions; undo whichever scheme this policy uses
    # (SmolVLA: MEAN_STD, pi05: QUANTILES) so Δ comes out in degrees either way.
    st = ds.meta.stats["action"]
    action_norm = str(cfg.normalization_mapping.get("ACTION", "MEAN_STD"))
    if "QUANTILE" in action_norm.upper():
        q01 = np.asarray(st["q01"], dtype=np.float64)[:6]
        q99 = np.asarray(st["q99"], dtype=np.float64)[:6]

        def unnormalize(x):  # inverse of 2*(x-q01)/(q99-q01) - 1
            return (x + 1.0) * (q99 - q01) / 2.0 + q01
    else:
        a_mean = np.asarray(st["mean"], dtype=np.float64)[:6]
        a_std = np.asarray(st["std"], dtype=np.float64)[:6]

        def unnormalize(x):
            return x * a_std + a_mean

    joints = list(ds.meta.features["action"]["names"])[:6]
    per_frame = []

    for p in picks:
        item = ds[starts[p]]
        base = {
            k: (v.unsqueeze(0).to(DEVICE) if isinstance(v, torch.Tensor) else v)
            for k, v in item.items()
            if isinstance(v, torch.Tensor)
        }
        chunks = []
        for prompt in PROMPTS:
            batch = dict(base)
            batch["task"] = [prompt]
            policy.reset()
            with torch.inference_mode():
                obs = preprocessor(batch)
                chunk = policy.predict_action_chunk(obs, noise=noise)
            raw = chunk.squeeze(0).float().cpu().numpy()[:, :6]
            chunks.append(unnormalize(raw))  # -> degrees

        # Per-joint mean |Δ| in degrees between the colour prompts, averaged over the chunk.
        per_joint = np.mean(
            [
                np.abs(chunks[i] - chunks[j]).mean(axis=0)
                for i in range(COLOURS)
                for j in range(i + 1, COLOURS)
            ],
            axis=0,
        )
        per_frame.append((item["task"], per_joint))

    mat = np.stack([d for _, d in per_frame])  # (frames, 6) degrees
    print(f"\ncheckpoint: {ckpt}")
    print(f"device: {DEVICE} | frames probed: {len(per_frame)}")
    print("\nmean |Δ| in DEGREES between the two colour prompts (same image, same noise):")
    for name, d in zip(joints, mat.mean(axis=0)):
        print(f"  {name:16s} {d:7.3f} deg")

    pan = float(mat.mean(axis=0)[0])
    print(f"\nshoulder_pan Δ = {pan:.3f} deg")
    print(f"switching pens requires a pan change of ~{SLOT_SEPARATION_DEG:.0f} deg,")
    print(f"so the instruction is moving {100 * pan / SLOT_SEPARATION_DEG:.1f}% of the distance needed.")
    if pan < 0.05 * SLOT_SEPARATION_DEG:
        print("VERDICT: instruction is ignored — the policy will pick the same pen regardless.")
    elif pan < 0.5 * SLOT_SEPARATION_DEG:
        print("VERDICT: partial — the instruction shifts the target but not far enough to switch pens.")
    else:
        print("VERDICT: the instruction genuinely retargets the arm. Worth a robot test.")


if __name__ == "__main__":
    main(
        sys.argv[1],
        int(sys.argv[2]) if len(sys.argv) > 2 else 12,
        sys.argv[3] if len(sys.argv) > 3 else REPO_ID,
    )
