"""Fine-tune SmolVLA on the `pick_pen` teleop dataset, on a rented Modal GPU.

Starts from the pretrained `lerobot/smolvla_base` (450M) and fine-tunes it on a dataset
recorded with `lerobot-record` (top + wrist cameras, three language-conditioned tasks:
"Pick up the blue pen" / "Pick up the pink pen" / "Pick up the grey pen").

Checkpoints land on a persistent Modal Volume, so a run that dies (or a timeout) can be
resumed instead of restarted. The final model is pushed to the Hub.

Prerequisites (one-time):
    uv tool install modal                              # puts `modal` on PATH, survives uv sync
    modal secret create wandb WANDB_API_KEY=xxx        # optional; then set USE_WANDB = True below
    # the HF token already lives in the existing `huggingface-secret`

Usage — the default spawns the job and returns immediately, so closing the terminal cannot
cancel it. Always pair with --detach:
    modal run --detach modal_train_pick_pen.py --fresh      # full run, A100-80GB, ~5h20m
    modal run --detach modal_train_pick_pen.py --resume     # continue from the last checkpoint
    modal run modal_train_pick_pen.py --steps 200 --no-push --tag smoke --wait   # watch a smoke test

Download checkpoints locally:
    modal volume get lerobot-outputs /smolvla_pick_pen ./outputs_from_modal
"""

import modal

REPO = "/Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1"
HF_USER = "bklassen3434"  # <-- set to your Hub username if different
# Default dataset; override per-run with `--dataset <stamped-name>` (lerobot-record appends a
# timestamp to the repo_id when it creates a dataset).
DATASET = f"{HF_USER}/pick_pen_20260919_165739"
MODEL_REPO = f"{HF_USER}/smolvla_pick_pen"
RUN_NAME = "smolvla_pick_pen"

# `lerobot/smolvla_base` bakes in three camera keys (camera1/2/3); our dataset records `top`
# and `wrist`. Map ours onto the first two — camera3 simply stays absent and SmolVLA skips it.
# The SAME map must be passed to `lerobot-rollout` at eval time.
RENAME_MAP = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}

# pi05 is loaded via --policy.type + --policy.pretrained_path, and in that path
# make_policy() fills input_features from the DATASET (factory.py: `if not cfg.input_features`).
# So the policy already expects `observation.images.top` / `.wrist` and renaming them to
# pi05_base's own names (base_0_rgb, ...) makes every image key "missing". No rename needed —
# the vision tower consumes whatever views are present, in feature order. Keep it empty at
# rollout time too, so train and eval agree on which camera fills which slot.
PI05_RENAME_MAP: dict[str, str] = {}

app = modal.App("lerobot-pick-pen")

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
    .run_commands("cd /root/lerobot && uv pip install --system -e '.[training,smolvla,peft,pi]'")
)

# Checkpoints survive across runs so training can be resumed.
outputs_vol = modal.Volume.from_name("lerobot-outputs", create_if_missing=True)
# HF cache: avoids re-downloading smolvla_base (~1.8GB) and the dataset on every run.
hf_cache_vol = modal.Volume.from_name("lerobot-hf-cache", create_if_missing=True)

# Set to True once `modal secret create wandb WANDB_API_KEY=...` exists. Worth doing before
# the long run — it's how you watch the loss curve with the laptop shut.
USE_WANDB = False

SECRETS = [modal.Secret.from_name("huggingface-secret")]
if USE_WANDB:
    SECRETS.append(modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"]))


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/root/lerobot/outputs": outputs_vol, "/root/.cache/huggingface": hf_cache_vol},
    secrets=SECRETS,
    timeout=60 * 60 * 12,  # 20k steps runs ~5h20m at batch 64 w/ unfrozen encoder — leave headroom
)
def train(
    steps: int = 20000,
    batch_size: int = 64,
    push: bool = True,
    resume: bool = False,
    tag: str = "",
    fresh: bool = False,
    dataset: str = "",
    lr: float = 0.0,
    freeze_vlm: bool = False,
    lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    policy: str = "smolvla",
    save_freq: int = 2500,
    contrastive: float = 0.0,
    contrastive_margin: float = 2.0,
    state_dropout: float = 0.0,
):
    import json
    import os
    import shutil
    import subprocess

    # The `huggingface-secret` may store the token under any of these names; huggingface_hub
    # only reads HF_TOKEN / HUGGING_FACE_HUB_TOKEN.
    if not os.environ.get("HF_TOKEN"):
        for alt in ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            if os.environ.get(alt):
                os.environ["HF_TOKEN"] = os.environ[alt]
                break
    print("HF token present:", bool(os.environ.get("HF_TOKEN")), flush=True)

    # LoRA only adapts the action expert's q/v projections plus the small action/state
    # projections (SmolVLA's built-in target list), so the VLM is untouched either way — but
    # freezing it too keeps it in eval() mode and makes the intent explicit.
    if lora:
        freeze_vlm = True

    # `tag` keeps throwaway runs (smoke tests) out of the real run's output dir — lerobot-train
    # refuses to start if its output_dir already exists.
    run_name = f"{RUN_NAME}_{tag}" if tag else RUN_NAME
    output_dir = f"outputs/{run_name}"

    if fresh and not resume:
        abs_dir = f"/root/lerobot/{output_dir}"
        if os.path.isdir(abs_dir):
            print(f"--fresh: deleting existing {output_dir}", flush=True)
            shutil.rmtree(abs_dir)
            outputs_vol.commit()

    is_pi05 = policy == "pi05"
    rename_map = PI05_RENAME_MAP if is_pi05 else RENAME_MAP

    # pi05_base's saved pipeline uses processor-step names this checkout doesn't register, so
    # route pi05 through a shim that adds the aliases before the trainer starts. See
    # my_contributions/tools/pi05_compat.py.
    # The contrastive term is a monkeypatch on SmolVLAPolicy.forward, so it has to be
    # imported before the trainer starts — same trick as the pi05 compat shim.
    shims = []
    if is_pi05:
        shims.append("import my_contributions.tools.pi05_compat;")
    if contrastive > 0:
        os.environ["CONTRASTIVE_LAMBDA"] = str(contrastive)
        os.environ["CONTRASTIVE_MARGIN"] = str(contrastive_margin)
        shims.append("import my_contributions.tools.contrastive_language_loss;")
    if state_dropout > 0:
        os.environ["STATE_DROPOUT_P"] = str(state_dropout)
        shims.append("import my_contributions.tools.state_dropout;")

    cmd = (
        [
            "python",
            "-c",
            "".join(shims) + "from lerobot.scripts.lerobot_train import main;main()",
        ]
        if shims
        else ["lerobot-train"]
    )
    if resume:
        # Resuming reads the policy + optimizer state from the checkpoint, so `--policy.path`
        # must NOT be passed here (it would take precedence and restart from the base model).
        cmd += [
            "--resume=true",
            f"--config_path={output_dir}/checkpoints/last/pretrained_model/train_config.json",
        ]
    elif is_pi05:
        # pi05 loads through --policy.type + --policy.pretrained_path rather than --policy.path.
        cmd += [
            "--policy.type=pi05",
            "--policy.pretrained_path=lerobot/pi05_base",
            "--policy.dtype=bfloat16",
            "--policy.gradient_checkpointing=true",
        ]
    else:
        cmd.append("--policy.path=lerobot/smolvla_base")  # the pretrained base, not from scratch

    cmd += [
        f"--dataset.repo_id={dataset or DATASET}",
        f"--rename_map={json.dumps(rename_map)}",
        "--policy.device=cuda",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        # Keep the LR schedule in sync with a shortened run (default decay is 30k steps).
        f"--policy.scheduler_decay_steps={steps}",
        # Unfreezing the vision encoder is a large win when specializing on a single task, but
        # it also lets finetuning destroy the VLM's language grounding. --freeze-vlm keeps
        # SmolVLA's defaults instead: only the action expert trains, so the language pathway
        # physically cannot be overwritten.
        f"--policy.freeze_vision_encoder={'true' if freeze_vlm else 'false'}",
        f"--policy.train_expert_only={'true' if freeze_vlm else 'false'}",
        # A lower LR than the 1e-4 default preserves more of the pretrained language grounding,
        # which full-strength finetuning on a small dataset can wipe out.
        *([f"--policy.optimizer_lr={lr}"] if lr else []),
        *(
            [
                "--peft.method_type=LORA",
                f"--peft.r={lora_r}",
                f"--peft.lora_alpha={lora_alpha}",
            ]
            if lora
            else []
        ),
        f"--save_freq={save_freq}",
        "--log_freq=100",
        f"--output_dir={output_dir}",
        f"--job_name={run_name}",
        f"--wandb.enable={'true' if USE_WANDB else 'false'}",
        f"--policy.push_to_hub={'true' if push else 'false'}",
    ]
    if push:
        cmd.append(f"--policy.repo_id={MODEL_REPO}{'_' + tag if tag else ''}")

    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd="/root/lerobot")
    outputs_vol.commit()  # persist checkpoints even if the run failed partway

    if proc.returncode != 0:
        raise RuntimeError(f"lerobot-train exited {proc.returncode}")
    print(f"DONE — checkpoints in volume lerobot-outputs:/{run_name}", flush=True)


@app.local_entrypoint()
def main(
    steps: int = 20000,
    batch_size: int = 64,
    push: bool = True,
    resume: bool = False,
    tag: str = "",
    fresh: bool = False,
    wait: bool = False,
    dataset: str = "",
    lr: float = 0.0,
    freeze_vlm: bool = False,
    lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    policy: str = "smolvla",
    save_freq: int = 2500,
    contrastive: float = 0.0,
    contrastive_margin: float = 2.0,
    state_dropout: float = 0.0,
):
    kwargs = dict(
        steps=steps,
        batch_size=batch_size,
        push=push,
        resume=resume,
        tag=tag,
        fresh=fresh,
        dataset=dataset,
        lr=lr,
        freeze_vlm=freeze_vlm,
        lora=lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        save_freq=save_freq,
        policy=policy,
        contrastive=contrastive,
        contrastive_margin=contrastive_margin,
        state_dropout=state_dropout,
    )
    if wait:
        # Blocking: streams output here. Good for short smoke tests you want to watch.
        train.remote(**kwargs)
        return

    # Fire-and-forget. spawn() leaves no client-side input handle, so closing the terminal
    # can't cancel the run the way a blocking .remote() can. Pair with `modal run --detach`.
    call = train.spawn(**kwargs)
    print(f"Spawned training, function call id: {call.object_id}")
    print("Safe to close this terminal. Follow along with:  modal app logs <app-id>")
