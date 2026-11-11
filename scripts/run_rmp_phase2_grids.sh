#!/usr/bin/env bash
# Phase 2 (E4) dictionary grids, one job at a time behind a vramorama lease (docs/rmp_phase2_plan.md).
# Usage: scripts/run_rmp_phase2_grids.sh <work-root>   e.g. /c/Users/patri/Documents/Codex/rmp_v0
set -u
ROOT="${1:?work root}"
WIN_ROOT="$(cygpath -w "$ROOT")"
PY="${PY:-/c/Users/patri/.venvs/bluebeam/Scripts/python.exe}"
VRAMORAMA="${VRAMORAMA:-/c/Users/patri/AppData/Local/Programs/vramorama/vramorama.exe}"
LOGS="$ROOT/phase2/logs"
mkdir -p "$LOGS"

grid() {
  local model="$1" kind="$2"; shift 2
  local log="$LOGS/${kind}_${model// /_}.log"
  echo "$(date -Iseconds) start $model $kind $*" >> "$LOGS/grids.log"
  "$VRAMORAMA" run --vram 3G --owner rmp-phase2-grid --quiet -- "$PY" -u -m research_gym.scripts.rmp_phase2_dictionaries \
    --model "$model" --kind "$kind" --artifact-dir "$WIN_ROOT\\artifacts" --monitor-bundle "$WIN_ROOT\\bundle" \
    --work-dir "$WIN_ROOT\\phase2\\dictionaries" "$@" >> "$log" 2>&1
  echo "$(date -Iseconds) done $model $kind exit $?" >> "$LOGS/grids.log"
}

grid "LOOP-T s1" shared
grid "LOOP-T s2" shared
grid "LOOP-T s3" shared
grid "FF-U s3" shared
grid "LOOP-T s1" trans
grid "FF-U s3" trans --sites 0 1 2 3 4 5 6 7
echo "$(date -Iseconds) grids finished" >> "$LOGS/grids.log"
