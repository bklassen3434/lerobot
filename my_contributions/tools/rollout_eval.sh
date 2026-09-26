#!/usr/bin/env bash
# Record one attempt per process, so the arm is stopped and parked while you reset the pens.
#
# lerobot-rollout's recording strategies (sentry, highlight) are built for continuous
# operation -- the policy keeps driving the arm between saved episodes, which is a nuisance
# when every attempt needs the pens moved by hand. Running sentry with --duration=<one attempt>
# sidesteps that: the loop exits on the duration limit, saves exactly one episode in its
# `finally` block, and `return_to_initial_position` parks the arm before the process ends.
# Cost is one policy load (~30-60 s) per attempt, which is worth it for a safe reset.
#
# Usage (from /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1):
#
#   ./my_contributions/tools/rollout_eval.sh <policy_dir> <task> <n_attempts> [stamped_repo_id]
#
# Leave the 4th argument off for the FIRST run of a session: lerobot-rollout stamps a
# timestamp onto the repo id at creation, and the script prints the stamped name at the end.
# Pass that name to every later run so all attempts land in one dataset.
#
#   ./my_contributions/tools/rollout_eval.sh outputs_from_modal/contrast_010000 blue 3
#   # -> prints  bklassen3434/rollout_pen_20260925_143000
#   ./my_contributions/tools/rollout_eval.sh outputs_from_modal/contrast_010000 pink 3 \
#        bklassen3434/rollout_pen_20260925_143000

set -euo pipefail

WORKSPACE=/Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1
cd "$WORKSPACE"

POLICY=${1:?policy dir, e.g. outputs_from_modal/contrast_010000}
TASK=${2:?bare colour word: blue or pink}
ATTEMPTS=${3:-3}
REPO=${4:-}

USER_ID=bklassen3434
CACHE=$HOME/.cache/huggingface/lerobot
DURATION=15            # seconds per attempt, matching the 15 s training episodes
FOLLOWER=/dev/tty.wchusbserial5B3E1213311
CAMERAS='{ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}'
# SmolVLA has camera1/camera2 baked into its config. This renames only the POLICY's inputs --
# the recorded dataset keeps top/wrist, which is what the progress model expects.
RENAME='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

echo "policy   : $POLICY"
echo "prompt   : \"$TASK\"   (bare colour word -- the contrastive models were trained on one-word labels)"
echo "attempts : $ATTEMPTS x ${DURATION}s"
echo

for i in $(seq 1 "$ATTEMPTS"); do
  echo "--------------------------------------------------------------------"
  echo "attempt $i/$ATTEMPTS   prompt: \"$TASK\""
  echo "Place both pens in the taped slots, then press Enter (Ctrl-C to stop)."
  read -r

  # A plain string, not an array: macOS ships bash 3.2, where expanding an empty array under
  # `set -u` aborts with "unbound variable". Unquoted, an empty RESUME_FLAG expands to nothing.
  # On resume, LeRobotDataset.resume() refuses to run without an explicit root (it would
  # otherwise write into the shared Hub snapshot cache). On creation the root must stay unset:
  # lerobot-rollout stamps a timestamp onto the repo id *after* config parsing, and the
  # directory has to follow the stamped name or nothing downstream can find it.
  if [ -n "$REPO" ]; then
    RESUME_FLAG="--resume=true --dataset.root=$CACHE/$REPO"
    CREATING=0
  else
    REPO="$USER_ID/rollout_pen"   # stamped with a timestamp on creation
    RESUME_FLAG=""
    CREATING=1
  fi

  # Don't let one bad attempt end the session. The Feetech bus on macOS intermittently drops a
  # status packet ("Failed to sync read 'Present_Position'"), which aborts the rollout process
  # -- but sentry's `finally` has already saved the episode, so the attempt is still usable.
  RC=0
  .venv/bin/python -c "import my_contributions.tools.motor_retry; from lerobot.scripts.lerobot_rollout import main; main()" \
    --strategy.type=sentry \
    --policy.path="$POLICY" \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER" \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="$CAMERAS" \
    --rename_map="$RENAME" \
    --dataset.repo_id="$REPO" \
    --dataset.single_task="$TASK" \
    --dataset.push_to_hub=false \
    --duration="$DURATION" \
    $RESUME_FLAG || RC=$?

  if [ "$CREATING" -eq 1 ]; then
    # Creation stamped a timestamp onto the name; read it back off disk for the next attempt.
    REPO="$USER_ID/$(ls -t "$CACHE/$USER_ID" | grep '^rollout_pen' | head -1)"
    echo "created dataset: $REPO"
  fi

  # A crash mid-rollout still leaves the episode on disk (sentry saves in its `finally`), but a
  # crash during setup does not. Count what actually landed rather than guessing.
  EPS=$(.venv/bin/python -c "
import glob, pyarrow.parquet as pq
print(sum(pq.read_metadata(f).num_rows for f in glob.glob('$CACHE/$REPO/meta/episodes/**/*.parquet', recursive=True)))
" 2>/dev/null || echo "?")
  if [ "$RC" -ne 0 ]; then
    echo ">>> attempt $i exited with status $RC"
  fi
  echo ">>> episodes in dataset so far: $EPS"
done

echo
echo "===================================================================="
echo "dataset: $REPO"
echo
echo "Grade it with:"
echo "  cd $WORKSPACE && uv run --no-sync --with matplotlib \\"
echo "    python -m my_contributions.tools.progress_model score $REPO --all-tasks"
