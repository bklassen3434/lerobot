"""Debug: run ONLY the FSDP config, streaming full output to see the real error."""

import modal

REPO = "/Users/benklassen/conductor/workspaces/lerobot/pattaya"
app = modal.App("lerobot-m4-fsdp-debug")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install("uv")
    .add_local_dir(
        REPO, "/root/lerobot", copy=True,
        ignore=[".git", ".venv", ".context", "outputs", "outputs_from_modal", "wandb",
                "tests/outputs", "__pycache__"],
    )
    .run_commands("cd /root/lerobot && uv pip install --system -e '.[training,smolvla]'")
)
vol = modal.Volume.from_name("lerobot-outputs", create_if_missing=True)


@app.function(image=image, gpu="A10G:2", volumes={"/root/lerobot/outputs": vol}, timeout=60 * 30)
def run():
    import subprocess

    cmd = [
        "accelerate", "launch", "--num_processes=2",
        "--use_fsdp",
        "--fsdp_sharding_strategy=FULL_SHARD",
        "--fsdp_auto_wrap_policy=NO_WRAP",
        "--fsdp_use_orig_params=true",
        "-m", "lerobot.scripts.lerobot_train",
        "--policy.type=smolvla", "--policy.device=cuda",
        "--dataset.repo_id=lerobot/pusht", "--dataset.episodes=[0,1,2]",
        "--batch_size=8", "--steps=30", "--save_checkpoint=false", "--save_freq=1000",
        "--log_freq=100", "--policy.push_to_hub=false", "--policy.use_amp=false",
        "--output_dir=outputs/m4/C_debug",
    ]
    # stream output straight to the Modal log (no capture) so we see the real traceback
    subprocess.run(cmd, cwd="/root/lerobot")


@app.local_entrypoint()
def main():
    run.remote()
