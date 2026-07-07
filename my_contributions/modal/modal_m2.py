"""M2 — Efficiency benchmark for SmolVLA on a rented Modal GPU.

Runs a small matrix of configs and prints a `M2_BENCH ...` line for each, so we can build the
before/after table: how mixed precision and gradient accumulation change peak GPU memory and
step time.

Configs:
  1. baseline            batch=8,  accum=1, fp32   -> the M1 baseline
  2. + mixed precision   batch=8,  accum=1, amp    -> should be faster / less memory
  3. big real batch      batch=32, accum=1, amp    -> effective batch 32 the naive way (memory cost)
  4. + grad accumulation batch=8,  accum=4, amp    -> effective batch 32 at ~batch-8 memory (the win)

Usage:
    modal run modal_m2.py
Then read the four `M2_BENCH` lines from the output.
"""

import modal

REPO = "/Users/benklassen/conductor/workspaces/lerobot/pattaya"

app = modal.App("lerobot-m2")

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
    .run_commands("cd /root/lerobot && uv pip install --system -e '.[training,smolvla]'")
)

vol = modal.Volume.from_name("lerobot-outputs", create_if_missing=True)

# (label, batch_size, grad_accum, use_amp)
CONFIGS = [
    ("1_baseline_fp32", 8, 1, False),
    ("2_amp", 8, 1, True),
    ("3_bigbatch32_amp", 32, 1, True),
    ("4_accum4_amp", 8, 4, True),
]


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/lerobot/outputs": vol},
    timeout=60 * 45,
)
def bench():
    import subprocess

    results = []
    for label, batch, accum, amp in CONFIGS:
        print(f"\n===== RUNNING CONFIG: {label} =====", flush=True)
        cmd = [
            "lerobot-train",
            "--policy.type=smolvla",
            "--policy.device=cuda",
            f"--policy.use_amp={'true' if amp else 'false'}",
            "--dataset.repo_id=lerobot/pusht",
            "--dataset.episodes=[0,1,2]",
            f"--batch_size={batch}",
            f"--gradient_accumulation_steps={accum}",
            "--steps=12",
            "--save_freq=100",  # >steps so we skip checkpoint writes during the benchmark
            "--save_checkpoint=false",
            "--log_freq=100",
            "--policy.push_to_hub=false",
            f"--output_dir=outputs/m2/{label}",
        ]
        # A failed config (e.g. OOM on the big batch) shouldn't abort the whole matrix.
        proc = subprocess.run(cmd, cwd="/root/lerobot", capture_output=True, text=True)
        bench_line = next(
            (ln for ln in proc.stdout.splitlines() + proc.stderr.splitlines() if "M2_BENCH" in ln),
            None,
        )
        if bench_line:
            print(f"RESULT {label}: {bench_line.split('M2_BENCH')[1].strip()}", flush=True)
            results.append((label, bench_line.split("M2_BENCH")[1].strip()))
        else:
            tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-8:])
            print(f"RESULT {label}: FAILED (exit {proc.returncode})\n{tail}", flush=True)
            results.append((label, f"FAILED (exit {proc.returncode})"))

    print("\n\n========== M2 SUMMARY ==========", flush=True)
    for label, r in results:
        print(f"{label:22s} {r}", flush=True)
    return results


@app.local_entrypoint()
def main():
    bench.remote()
