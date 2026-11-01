#!/usr/bin/env bash
# AITK_JOB_LAUNCHER for the ai-toolkit UI on Spark. The UI worker runs this in place of
# `python run.py <config>`; it starts the job through train_spark.sh (gateway GPU claim,
# OOM shield, supervisor that releases the claim) in the training container, then stays in
# the foreground for the worker: the container log streams to stdout (the job's log.txt)
# and we exit with the container's exit code.
#
#   ui_job_spark.sh <run.py> <config.json>      (args exactly as the worker passes them)
#
# Stop from the UI: the worker sets the job's stop flag in the sqlite db, which the trainer
# polls and shuts down on cleanly, then SIGINTs us. We keep waiting for that clean exit and
# only `docker stop` the container if it is still up STOP_GRACE_S later.
#   AITK_CLAIM_GB   GPU claim size (default 50)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
CONFIG="${2:?usage: ui_job_spark.sh <run.py> <config.json>}"
CONTAINER=ai-toolkit-training
STOP_GRACE_S="${STOP_GRACE_S:-120}"

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)" = "true" ]; then
  echo "$CONTAINER is already running another job; one training job at a time on Spark." >&2
  exit 1
fi

./train_spark.sh "$CONFIG" - "${AITK_CLAIM_GB:-50}"
# Pin to this job's container: the name is reused by the next job.
CID="$(docker inspect -f '{{.Id}}' "$CONTAINER")"

backstop=""
on_stop() {
  echo "==> stop requested; waiting up to ${STOP_GRACE_S}s for the trainer to exit cleanly"
  [ -n "$backstop" ] && return
  ( sleep "$STOP_GRACE_S"; docker stop "$CID" >/dev/null 2>&1 ) &
  backstop=$!
}
trap on_stop INT TERM

docker logs -f --since "$(docker inspect -f '{{.State.StartedAt}}' "$CID")" "$CID" 2>&1 &
logs=$!
docker wait "$CID" > "${TMPDIR:-/tmp}/aitk_wait.$$" &
waiter=$!
# `wait` returns early when a trapped signal arrives; keep waiting for the container.
until wait "$waiter"; do kill -0 "$waiter" 2>/dev/null || break; done
code="$(cat "${TMPDIR:-/tmp}/aitk_wait.$$" 2>/dev/null || echo 1)"
rm -f "${TMPDIR:-/tmp}/aitk_wait.$$"
kill "$logs" 2>/dev/null || true
if [ -n "$backstop" ]; then
  pkill -P "$backstop" 2>/dev/null || true
  kill "$backstop" 2>/dev/null || true
fi
exit "${code:-1}"
