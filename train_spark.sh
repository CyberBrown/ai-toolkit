#!/usr/bin/env bash
# train_spark.sh — launch an ai-toolkit training run on Spark that survives the night.
#
# Canonical copy lives in dhm-app at packages/raw-process/scripts/spark/train_spark.sh;
# deployed copy on prometheus is ~/projects/ai-toolkit/train_spark.sh. Edit here first.
#
# ── Why this script exists ────────────────────────────────────────────────────
# A GPU claim alone does NOT make a long run survivable. LL-5bf2a798 (Qwen indoor
# LoRA) held a correct 80 GB claim for four hours and was still SIGKILLed at step
# 267/500, because three things are true at once on prometheus:
#
#   * `ollama` was registered with `gateway.vram=0`, so **no claim constrained it**.
#     An inbound chat request loaded a model regardless of what was training.
#     FIXED 2026-08-06: the ollama container now declares `gateway.vram="64"`, so a
#     training-sized claim leaves too little headroom for the scheduler to start it.
#   * The gateway models **VRAM only**. GB10's 128 GB is unified, so when the GPU
#     is full ollama offloads to CPU (`offloaded 1/33 layers to GPU`) and the
#     pressure lands in host RAM — the one dimension nothing watches.
#   * `earlyoom` is the actual arbiter (floors: mem 5 % / swap 10 %), it picks the
#     largest-RSS process, and it knows nothing about gateway claims.
#
# See wiki/DECISIONS.md D-032 and wiki/2026-08-06-qwen-indoor-lora-pilot.md.
#
# ── What it does ──────────────────────────────────────────────────────────────
#   1. GPU claim    — gateway stops starting registered GPU services into the
#                     reserved headroom and evicts running ones. Container-tagged,
#                     so the gateway auto-releases it when training exits.
#   2. Quiesce      — stops the consumers the claim system does not model
#                     (default: ollama), and remembers which were running.
#   3. OOM shield   — lowers the training process's OOM badness so earlyoom does
#                     not pick it first.
#   4. Supervisor   — a detached watcher keeps the shield applied to new pids and
#                     restores every quiesced service when the container exits.
#
# Usage: ./train_spark.sh <config.yaml> <dataset_dir> [vram_gb]
#
# Env:
#   TRAIN_QUIESCE    space-separated gateway service names to stop for the run's
#                    lifetime. Default "ollama". Set to "" to disable quiescing.
#                    NB `ollama-embed` is deliberately NOT in this list. An external
#                    host (192.168.11.31, the metis RAG batch) POSTs
#                    /services/ollama-embed/start every 15 minutes on the quarter
#                    hour, so quiescing it is theatre — it returns within 15 min.
#                    It is also honestly declared (gateway.vram=3, ~2 GB actual), so
#                    it is accounted for rather than invisible. If a run genuinely
#                    needs a bare box, pause that cron on 192.168.11.31 instead.
#   TRAIN_OOM_ADJ    oom_score_adj applied to training pids. Default -800.
#                    Range -1000..0; -1000 makes it effectively unkillable by
#                    earlyoom, which then kills something else instead.
#   TRAIN_NO_SHIELD  set to 1 to skip the OOM shield.
#   GATEWAY_URL      default http://127.0.0.1:8080
#   GATEWAY_API_TOKEN  read from ~/projects/gateway/.env if unset.
#
# How the two halves fit together:
#   The `gateway.vram="64"` label on the ollama container is what stops the *scheduler*
#   starting it into a training claim's headroom. Quiescing here is what stops a model
#   that is *already loaded* — and it is the only protection against a request sent
#   straight to ollama's port 11434, which bypasses the gateway entirely. Keep both.
#
# Still open (a bigger lever than either): OLLAMA_NUM_PARALLEL=4 x 131072 ctx makes a
#   single llama3.1:8b chat allocate ~77 GB. Lowering parallelism or capping context
#   would shrink the blast radius far more than any accounting fix. Not changed here
#   because it alters chat behaviour — Chris's call.

set -euo pipefail

CONTAINER="ai-toolkit-training"
COMPOSE_FILE="docker-compose.ai-toolkit.yml"
GW="${GATEWAY_URL:-http://127.0.0.1:8080}"
QUIESCE="${TRAIN_QUIESCE-ollama}"
OOM_ADJ="${TRAIN_OOM_ADJ:--800}"
STATE_DIR="${XDG_RUNTIME_DIR:-/tmp}/train_spark"
LOG="$STATE_DIR/supervisor.log"

TOKEN="${GATEWAY_API_TOKEN:-$(grep -oP '(?<=^GATEWAY_API_TOKEN=).*' "$HOME/projects/gateway/.env" 2>/dev/null || true)}"
auth=(); [ -n "$TOKEN" ] && auth=(-H "Authorization: Bearer $TOKEN")

# -f so a non-2xx is a non-zero exit, not a silently-swallowed error body.
gw() {  # gw <method> <path> [data]
  local method="$1" path="$2" data="${3:-}"
  if [ -n "$data" ]; then
    curl -fsS -m 15 "${auth[@]}" -X "$method" "$GW$path" -H 'Content-Type: application/json' -d "$data"
  else
    curl -fsS -m 15 "${auth[@]}" -X "$method" "$GW$path"
  fi
}

# Host pids inside a container, via its systemd cgroup. NB `docker top <c> -eo pid=`
# does NOT work on this box ("Couldn't find PID field in ps output") — don't switch back.
container_pids() {
  local cid="$1" f="/sys/fs/cgroup/system.slice/docker-$1.scope/cgroup.procs"
  [ -r "$f" ] && cat "$f" 2>/dev/null || true
}

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }

# ── supervisor mode ───────────────────────────────────────────────────────────
# Re-entry point: train_spark.sh --supervise <container> <claim_id> <restore...>
if [ "${1:-}" = "--supervise" ]; then
  shift
  sup_container="$1"; sup_claim="$2"; shift 2
  sup_restore=("$@")

  shield() {
    [ "${TRAIN_NO_SHIELD:-0}" = "1" ] && return 0
    local p n=0
    for p in $(container_pids "$sup_container"); do
      if echo "$OOM_ADJ" | sudo -n tee "/proc/$p/oom_score_adj" >/dev/null 2>&1; then
        n=$((n + 1))
      fi
    done
    [ "$n" -gt 0 ] && [ "${sup_shielded:-0}" = "0" ] && { log "shield applied to $n pid(s)"; sup_shielded=1; }
    return 0
  }
  sup_shielded=0

  log "supervisor up — container=$sup_container claim=$sup_claim restore='${sup_restore[*]:-none}' oom_adj=$OOM_ADJ"
  while [ "$(docker inspect -f '{{.State.Running}}' "$sup_container" 2>/dev/null || echo false)" = "true" ]; do
    shield
    sleep 20
  done

  code="$(docker inspect -f '{{.State.ExitCode}}' "$sup_container" 2>/dev/null || echo '?')"
  log "container exited (code=$code) — restoring"

  for svc in "${sup_restore[@]:-}"; do
    [ -z "$svc" ] && continue
    # no /enable — label-driven registry, returns 405. `start` is the lever.
    if ! gw POST "/services/$svc/start" >/dev/null 2>&1; then
      docker start "$svc" >/dev/null 2>&1 || log "WARN could not restart $svc — do it by hand"
    fi
    log "restored $svc"
  done

  if [ -n "$sup_claim" ] && [ "$sup_claim" != "-" ]; then
    gw DELETE "/gpu/claim/$sup_claim" >/dev/null 2>&1 || true
  fi
  log "supervisor done"
  exit 0
fi

# ── normal mode ───────────────────────────────────────────────────────────────
CONFIG="${1:?usage: train_spark.sh <config.yaml> <dataset_dir> [vram_gb]}"
DATASET="${2:?dataset dir required (or - when the config names its own dataset paths, as UI jobs do)}"
VRAM_GB="${3:-48}"

[ -f "$COMPOSE_FILE" ] || { echo "run me from the ai-toolkit dir (no $COMPOSE_FILE here)" >&2; exit 1; }
[ -f "$CONFIG" ]  || { echo "config not found: $CONFIG"   >&2; exit 1; }
[ "$DATASET" = "-" ] && DATASET=./datasets
[ -d "$DATASET" ] || { echo "dataset not found: $DATASET" >&2; exit 1; }

mkdir -p "$STATE_DIR"

if ! sudo -n true 2>/dev/null && [ "${TRAIN_NO_SHIELD:-0}" != "1" ]; then
  echo "WARN: no passwordless sudo — the OOM shield will be best-effort only." >&2
fi

# 1. quiesce the unaccounted consumers, remembering what to put back
RESTORE=()
for svc in $QUIESCE; do
  running="$(docker inspect -f '{{.State.Running}}' "$svc" 2>/dev/null || echo false)"
  echo "==> quiescing $svc (running=$running)"
  # NB do NOT call /services/<name>/disable — the gateway registry is label-driven and
  # returns 405 ("edit its gateway.* labels ... and recreate it"). `stop` is the only
  # lever that works. What stops the scheduler restarting it mid-run is the service's
  # declared gateway.vram vs the claim's headroom, which is why the ollama container
  # carries gateway.vram="64" (see D-032) — that label and this stop work as a pair.
  if ! gw POST "/services/$svc/stop" >/dev/null 2>&1; then
    docker stop "$svc" >/dev/null 2>&1 || echo "    WARN could not stop $svc" >&2
  fi
  # NB `[ ... ] && RESTORE+=(...)` would abort the script under `set -e` whenever a
  # quiesce target is already stopped (failing last command in the loop body).
  if [ "$running" = "true" ]; then RESTORE+=("$svc"); fi
done

restore_now() {  # only used if we fail before the supervisor is up
  for svc in "${RESTORE[@]:-}"; do
    [ -z "$svc" ] && continue
    gw POST "/services/$svc/start" >/dev/null 2>&1 || docker start "$svc" >/dev/null 2>&1 || true
  done
}
trap 'restore_now' ERR

# 2. start training
echo "==> starting training container"
DATASET_PATH="$DATASET" CONFIG_FILE="$CONFIG" \
  docker compose -f "$COMPOSE_FILE" up -d train

CID="$(docker inspect -f '{{.Id}}' "$CONTAINER")"

# 3. place the GPU claim
echo "==> placing GPU claim (${VRAM_GB}GB, priority 0) on $CONTAINER"
resp="$(gw POST /gpu/claim \
  "{\"name\":\"training\",\"vram_gb\":$VRAM_GB,\"container\":\"$CONTAINER\",\"priority\":0}")"
echo "    $resp"
CLAIM_ID="$(printf '%s' "$resp" | grep -oP '(?<="id":")[^"]+' | head -1)"
[ -n "$CLAIM_ID" ] || CLAIM_ID="-"

trap - ERR

# 4. hand off to the detached supervisor
setsid nohup "$(readlink -f "$0")" --supervise "$CID" "$CLAIM_ID" "${RESTORE[@]:-}" \
  >>"$LOG" 2>&1 < /dev/null &
disown || true

cat <<EOF

Training is running with all three protections in place.
  - GPU claim ${VRAM_GB}GB P0 (id=$CLAIM_ID) — registered GPU services blocked/evicted
  - quiesced: ${QUIESCE:-none}   (restored automatically on exit: ${RESTORE[*]:-none})
  - OOM shield: oom_score_adj=$OOM_ADJ on training pids, reapplied every 20s
  - supervisor log: $LOG

  watch:   docker logs -f $CONTAINER
  claims:  curl -s ${auth[*]:+-H 'Authorization: Bearer ***'} $GW/gpu/claims
  abort:   docker stop $CONTAINER   (supervisor restores everything)

NOTE: quiescing ollama means chat/RAG on this box is DOWN for the run's duration.
      That is the point — an unaccounted ollama model load is what killed
      LL-5bf2a798. The durable fix is a realistic gateway.vram on the ollama
      container (see the header comment).
EOF
