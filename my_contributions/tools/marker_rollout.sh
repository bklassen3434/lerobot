#!/usr/bin/env bash
# Run the marker model on the real arm: Claude picks the colour, the detector rings that pen,
# SmolVLA picks up the ringed pen. One attempt per process, like rollout_eval.sh (whose
# comments explain the sentry/--duration/resume tricks reused here).
#
# Per attempt:
#   1. you place the pens and type an instruction (Enter = reuse the last one)
#   2. marker_aim.py: Claude -> colour, top camera -> ring position, preview image opens
#   3. you check the ring is on the right pen and press Enter (Ctrl-C aborts, arm never moves)
#   4. lerobot-rollout with marker_shim.py drawing that ring on every top frame
#
# Usage (from /Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1):
#   ./my_contributions/tools/marker_rollout.sh "pick up the pink pen" 3
#   ./my_contributions/tools/marker_rollout.sh "pick up the blue pen" 3 bklassen3434/rollout_marker_<stamp>

set -euo pipefail

WORKSPACE=/Users/benklassen/conductor/workspaces/lerobot/hyderabad-v1
cd "$WORKSPACE"

INSTRUCTION=${1:?instruction, e.g. "pick up the pink pen"}
ATTEMPTS=${2:-3}
REPO=${3:-}
POLICY=${POLICY:-outputs_from_modal/smolvla_pick_pen_v2_marked/checkpoints/010000/pretrained_model}
TASK="pick up the marked pen"   # the constant prompt the marker model was trained on

USER_ID=bklassen3434
CACHE=$HOME/.cache/huggingface/lerobot
DURATION=15
FOLLOWER=/dev/tty.wchusbserial5B3E1213311
CAMERAS='{ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}'
RENAME='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
PREVIEW=my_contributions/agent/out/marker_aim.jpg

# Episodes saved so far in the dataset (0 before it exists).
count_episodes() {
  [ -n "$REPO" ] && [ -d "$CACHE/$REPO/meta/episodes" ] || { echo 0; return; }
  .venv/bin/python -c "
import glob, pyarrow.parquet as pq
print(sum(pq.read_metadata(f).num_rows for f in glob.glob('$CACHE/$REPO/meta/episodes/**/*.parquet', recursive=True)))
" 2>/dev/null || echo 0
}

echo "policy   : $POLICY"
echo "attempts : $ATTEMPTS x ${DURATION}s"
echo

for i in $(seq 1 "$ATTEMPTS"); do
  echo "--------------------------------------------------------------------"
  echo "attempt $i/$ATTEMPTS"
  echo "Park the arm, place both pens, then type an instruction"
  read -r -p "  (Enter = \"$INSTRUCTION\"): " NEW
  [ -n "$NEW" ] && INSTRUCTION=$NEW

  if ! AIM=$(uv run --no-sync python my_contributions/tools/marker_aim.py "$INSTRUCTION"); then
    echo ">>> aiming failed, attempt $i skipped (the arm did not move)"
    continue
  fi
  UV=$(echo "$AIM" | sed -n 's/^MARKER_UV=//p')
  COLOUR=$(echo "$AIM" | sed -n 's/^COLOUR=//p')
  open "$PREVIEW"
  read -r -p "Is the green ring on the right pen? Enter = run the arm, Ctrl-C = abort: "

  if [ -n "$REPO" ]; then
    RESUME_FLAG="--resume=true --dataset.root=$CACHE/$REPO"
    CREATING=0
  else
    REPO="$USER_ID/rollout_marker"
    RESUME_FLAG=""
    CREATING=1
  fi

  EPS_BEFORE=$(count_episodes)
  RC=0
  MARKER_UV="$UV" .venv/bin/python -c "import my_contributions.tools.motor_retry, my_contributions.tools.marker_shim; from lerobot.scripts.lerobot_rollout import main; main()" \
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
    REPO="$USER_ID/$(ls -t "$CACHE/$USER_ID" | grep '^rollout_marker' | head -1)"
    echo "created dataset: $REPO"
  fi
  [ "$RC" -ne 0 ] && echo ">>> attempt $i exited with status $RC"

  # Log what was asked and how it went, next to the episode, for make_demo_video.py.
  EPS_AFTER=$(count_episodes)
  if [ "$EPS_AFTER" -gt "$EPS_BEFORE" ]; then
    read -r -p "Did it go to the $COLOUR pen? [y/n]: " RIGHT
    read -r -p "Did it lift the pen? [y/n]: " LIFTED
    printf '%s\t%s\t%s\t%s\t%s\n' "$((EPS_AFTER - 1))" "$INSTRUCTION" "$COLOUR" "${RIGHT:-?}" "${LIFTED:-?}" \
      >> "$CACHE/$REPO/meta/instructions.tsv"
    echo ">>> logged episode $((EPS_AFTER - 1)): \"$INSTRUCTION\" -> $COLOUR, right pen: ${RIGHT:-?}, lifted: ${LIFTED:-?}"
  else
    echo ">>> no episode was saved for attempt $i"
  fi
done

echo
echo "===================================================================="
echo "dataset: $REPO   (every attempt recorded, ring included)"
echo
echo "Make the demo video with:"
echo "  cd $WORKSPACE && uv run --no-sync python my_contributions/tools/make_demo_video.py $REPO"
