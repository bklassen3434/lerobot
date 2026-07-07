"""M4 — Distributed training: 1-GPU vs DDP vs FSDP for SmolVLA on Modal.

Runs four configs on a 2x A10G container and reports peak per-GPU memory + throughput, so we can
show data parallelism (DDP) and sharded training (FSDP):

  A0  1 GPU,  fp32              -> baseline
  A1  1 GPU,  bf16 (real AMP)   -> isolates mixed precision (corrects the M2 noise result)
  B   2 GPU,  DDP,  fp32        -> data parallelism: ~2x throughput, full model per GPU
  C   2 GPU,  FSDP, fp32        -> shards model+grads+optimizer: lower per-GPU memory

Launchers:
  A0/A1 : plain `lerobot-train`
  B     : `torchrun --nproc_per_node=2` (Accelerate auto-detects DDP)
  C     : `accelerate launch --use_fsdp ...`

Usage:
    modal run modal_m4.py
"""

import modal

REPO = "/Users/benklassen/conductor/workspaces/lerobot/pattaya"

app = modal.App("lerobot-m4")

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

COMMON = [
    "--policy.type=smolvla",
    "--policy.device=cuda",
    "--dataset.repo_id=lerobot/pusht",
    "--dataset.episodes=[0,1,2]",
    "--batch_size=8",
    "--steps=30",
    "--save_checkpoint=false",
    "--save_freq=1000",
    "--log_freq=100",
    "--policy.push_to_hub=false",
]

MODULE = ["-m", "lerobot.scripts.lerobot_train"]

FSDP_FLAGS = [
    "--use_fsdp",
    "--fsdp_sharding_strategy=FULL_SHARD",
    # NO_WRAP: treat the whole model as one FSDP unit. SmolVLA's forward accesses some weights
    # outside their owning submodule, so per-layer (SIZE/TRANSFORMER) wrapping breaks it; NO_WRAP
    # gathers the full model each step, which works but limits FSDP's memory benefit.
    "--fsdp_auto_wrap_policy=NO_WRAP",
    "--fsdp_use_orig_params=true",
]


def cfg_cmd(label, amp):
    return ["lerobot-train", *COMMON, f"--policy.use_amp={'true' if amp else 'false'}",
            f"--output_dir=outputs/m4/{label}"]


def ddp_cmd(label, amp):
    return ["torchrun", "--nproc_per_node=2", *MODULE, *COMMON,
            f"--policy.use_amp={'true' if amp else 'false'}", f"--output_dir=outputs/m4/{label}"]


def fsdp_cmd(label, amp):
    return ["accelerate", "launch", "--num_processes=2", *FSDP_FLAGS, *MODULE, *COMMON,
            f"--policy.use_amp={'true' if amp else 'false'}", f"--output_dir=outputs/m4/{label}"]


# (label, command)
CONFIGS = [
    ("A0_1gpu_fp32", cfg_cmd("A0_1gpu_fp32", amp=False)),
    ("A1_1gpu_bf16", cfg_cmd("A1_1gpu_bf16", amp=True)),
    ("B_2gpu_ddp_fp32", ddp_cmd("B_2gpu_ddp_fp32", amp=False)),
    ("C_2gpu_fsdp_fp32", fsdp_cmd("C_2gpu_fsdp_fp32", amp=False)),
]


@app.function(
    image=image,
    gpu="A10G:2",  # two GPUs for the DDP / FSDP configs
    volumes={"/root/lerobot/outputs": vol},
    timeout=60 * 60,
)
def bench():
    import re
    import subprocess

    results = []
    for label, cmd in CONFIGS:
        print(f"\n===== RUNNING CONFIG: {label} =====\n{' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd="/root/lerobot", capture_output=True, text=True)
        out = proc.stdout + "\n" + proc.stderr
        bench_line = next((ln for ln in out.splitlines() if "M2_BENCH" in ln), "")
        mem = re.search(r"peak_gpu_mem_mb=(\d+)", bench_line)
        step_ms = re.search(r"avg_step_ms=([\d.]+)", bench_line)
        eff = re.search(r"effective_batch=(\d+)", bench_line)
        if mem and step_ms and eff:
            mem, step_ms, eff = int(mem.group(1)), float(step_ms.group(1)), int(eff.group(1))
            samples_s = 1000 * eff / step_ms
            summary = (
                f"peak_per_gpu_mem={mem}MB | avg_step={step_ms:.0f}ms | "
                f"effective_batch={eff} | throughput={samples_s:.1f} samples/s"
            )
        else:
            tail = "\n".join(out.splitlines()[-12:])
            summary = f"FAILED (exit {proc.returncode})\n{tail}"
        print(f"RESULT {label}: {summary}", flush=True)
        results.append((label, summary))

    print("\n\n========== M4 SUMMARY ==========", flush=True)
    for label, r in results:
        print(f"{label:20s} {r}", flush=True)
    return results


@app.local_entrypoint()
def main():
    bench.remote()
