#!/usr/bin/env bash
# Runs RMP jobs from a queue file, one at a time, each GPU job behind a vramorama lease that stops it while
# the card is power-capped and restarts it afterwards (every job here resumes or skips finished work).
# Lines are shell commands using the functions in scripts/rmp_queue_functions.sh; blank lines and lines
# starting with # are skipped. The queue file is reread before every line, so lines can be added or
# inserted while the runner goes. Finished lines are recorded in <queue>.done, so rerunning the runner
# resumes. The first failing line stops the runner (recorded in <queue>.failed) so that nothing downstream
# runs on a bad input. When the queue is exhausted the runner waits up to an hour for more lines, then exits.
# Usage: scripts/run_rmp_queue.sh <work-root> <queue-file>   e.g. /c/Users/patri/Documents/Codex/rmp_v0 ...
set -u
ROOT="${1:?work root}"
QUEUE="${2:?queue file}"
WIN_ROOT="$(cygpath -w "$ROOT")"
PY="${PY:-/c/Users/patri/.venvs/bluebeam/Scripts/python.exe}"
VRAMORAMA="${VRAMORAMA:-/c/Users/patri/AppData/Local/Programs/vramorama/vramorama.exe}"
LOG="${QUEUE%.*}.log"
DONE="$QUEUE.done"
touch "$DONE"
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=rmp_queue_functions.sh
source scripts/rmp_queue_functions.sh

idle=0
note "runner started on $QUEUE"
while true; do
  next=""
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in ''|'#'*) continue ;; esac
    grep -Fxq -- "$line" "$DONE" || { next="$line"; break; }
  done < "$QUEUE"
  if [ -z "$next" ]; then
    idle=$((idle + 1))
    [ "$idle" -gt "${IDLE_MINUTES:-60}" ] && { note "queue exhausted; runner exits"; exit 0; }
    sleep 60
    continue
  fi
  idle=0
  wait_host
  note "start: $next"
  eval "$next" >> "$LOG" 2>&1
  status=$?
  if [ "$status" -ne 0 ]; then
    note "FAILED ($status): $next"
    echo "$next" >> "$QUEUE.failed"
    exit "$status"
  fi
  echo "$next" >> "$DONE"
  note "done: $next"
done
