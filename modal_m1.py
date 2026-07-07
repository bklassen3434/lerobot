"""M1 — Baseline + profile SmolVLA on a rented Modal GPU.

Usage:
    pip install modal          # once
    modal setup                # once, links your Modal account
    modal run modal_m1.py      # runs the profiled training job on an A10G GPU

The job trains SmolVLA for a few steps with the profiler enabled (LEROBOT_PROFILE=1),
writes the trace to a persistent Modal Volume, so you can download and inspect it
after the ephemeral GPU container shuts down.

Download the trace afterwards with:
    modal volume get lerobot-outputs m1/ ./outputs_from_modal
Then open the *.pt.trace.json in chrome://tracing, or:
    tensorboard --logdir ./outputs_from_modal/m1/profiler
"""

import modal

# This bundles YOUR local edited repo (including the M1 profiler changes) into the image.
REPO = "/Users/benklassen/conductor/workspaces/lerobot/pattaya"

app = modal.App("lerobot-m1")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install("uv")
    # copy the local repo in (ignore heavy/irrelevant dirs to keep the image small)
    .add_local_dir(
        REPO,
        "/root/lerobot",
        copy=True,
        ignore=[
            ".git",
            ".venv",
            ".context",
            "outputs",
            "outputs_from_modal",
            "wandb",
            "tests/outputs",
            "__pycache__",
        ],
    )
    .run_commands("cd /root/lerobot && uv pip install --system -e '.[training,smolvla]'")
)

# Persistent storage so checkpoints + profiler traces survive after the container exits.
vol = modal.Volume.from_name("lerobot-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # cheap 24GB GPU; plenty for SmolVLA at these step counts
    volumes={"/root/lerobot/outputs": vol},
    # If you hit HuggingFace rate limits/auth, create a Modal secret named "huggingface"
    # (modal secret create huggingface HF_TOKEN=hf_xxx) and uncomment the next line:
    # secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 30,
)
def train():
    import os
    import subprocess

    env = {**os.environ, "LEROBOT_PROFILE": "1"}  # <-- turns on the M1 profiler
    subprocess.run(
        [
            "lerobot-train",
            # Fresh SmolVLA sized to the dataset. For M1 we're profiling the compute, so random
            # vs. pretrained weights give an identical performance profile. (M3/LoRA will instead
            # load the pretrained `lerobot/smolvla_base` against a camera-matched dataset.)
            "--policy.type=smolvla",
            "--policy.device=cuda",
            "--dataset.repo_id=lerobot/pusht",
            "--dataset.episodes=[0,1,2]",
            "--batch_size=8",
            "--steps=10",  # only need ~10 steps to capture the profile
            "--save_freq=10",
            "--policy.push_to_hub=false",  # local profiling run; don't push to the Hub
            "--output_dir=outputs/m1/baseline",
        ],
        cwd="/root/lerobot",
        env=env,
        check=True,
    )
    vol.commit()  # persist outputs (checkpoints + profiler trace) to the Volume


@app.local_entrypoint()
def main():
    train.remote()
