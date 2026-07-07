"""M3 — LoRA fine-tuning of the PRETRAINED SmolVLA on a rented Modal GPU.

Loads the real `lerobot/smolvla_base` foundation model (with its learned weights) and fine-tunes it
two ways, to show what LoRA buys:

  1. full_finetune : normal fine-tuning (SmolVLA already freezes its vision encoder, so this trains
                     the action-expert — ~100M params).
  2. lora_r16      : LoRA adapters only — trains a tiny fraction of the parameters.

`--policy.input_features=null --policy.output_features=null` lets the pretrained base re-infer its
inputs from the (single-camera) pusht dataset, so no camera-matched dataset is needed. This mirrors
the tested command in tests/test_cli_peft.py.

Usage:
    modal run modal_m3.py
Then read the RESULT lines (learnable vs total params, loss, memory).
"""

import modal

REPO = "/Users/benklassen/conductor/workspaces/lerobot/pattaya"

app = modal.App("lerobot-m3")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install("uv")
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
    .run_commands("cd /root/lerobot && uv pip install --system -e '.[training,smolvla,peft]'")
)

vol = modal.Volume.from_name("lerobot-outputs", create_if_missing=True)

# (label, use_lora)
CONFIGS = [
    ("full_finetune", False),
    ("lora_r16", True),
]


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/lerobot/outputs": vol},
    timeout=60 * 45,
)
def bench():
    import re
    import subprocess

    results = []
    for label, use_lora in CONFIGS:
        print(f"\n===== RUNNING CONFIG: {label} =====", flush=True)
        cmd = [
            "lerobot-train",
            "--policy.path=lerobot/smolvla_base",  # the REAL pretrained foundation model
            "--policy.push_to_hub=false",
            "--policy.input_features=null",  # re-infer inputs from the dataset
            "--policy.output_features=null",
            "--policy.device=cuda",
            "--policy.use_amp=true",
            "--dataset.repo_id=lerobot/pusht",
            "--dataset.episodes=[0,1,2]",
            "--batch_size=8",
            "--steps=60",
            "--save_checkpoint=false",
            "--save_freq=1000",
            "--log_freq=10",
            f"--output_dir=outputs/m3/{label}",
        ]
        if use_lora:
            cmd += ["--peft.method=LORA", "--peft.r=16", "--peft.lora_alpha=32"]

        proc = subprocess.run(cmd, cwd="/root/lerobot", capture_output=True, text=True)
        out = proc.stdout + "\n" + proc.stderr

        def grab(pattern, default="?"):
            m = re.search(pattern, out)
            return m.group(1) if m else default

        learnable = grab(r"num_learnable_params=(\d+)")
        total = grab(r"num_total_params=(\d+)")
        bench_line = next((ln for ln in out.splitlines() if "M2_BENCH" in ln), "")
        mem = grab(r"peak_gpu_mem_mb=(\d+)")
        # first and last logged loss
        losses = re.findall(r"loss:([\d.]+)", out)
        first_loss = losses[0] if losses else "?"
        last_loss = losses[-1] if losses else "?"

        if learnable != "?" and total != "?":
            pct = 100 * int(learnable) / int(total)
            summary = (
                f"learnable={int(learnable):,} / total={int(total):,} ({pct:.2f}% trainable) | "
                f"loss {first_loss}->{last_loss} | peak_mem={mem}MB"
            )
        else:
            tail = "\n".join(out.splitlines()[-10:])
            summary = f"FAILED (exit {proc.returncode})\n{tail}"
        print(f"RESULT {label}: {summary}", flush=True)
        results.append((label, summary))

    print("\n\n========== M3 SUMMARY ==========", flush=True)
    for label, r in results:
        print(f"{label:16s} {r}", flush=True)
    return results


@app.local_entrypoint()
def main():
    bench.remote()
