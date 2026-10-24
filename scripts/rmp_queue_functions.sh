# Functions for RMP queue lines (scripts/run_rmp_queue.sh). Kept apart from the runner so that a queue line
# `source scripts/rmp_queue_functions.sh` updates them in a runner that is already going.
# Expects ROOT, WIN_ROOT, PY, VRAMORAMA and LOG from the runner.

FAMILIES="pointer_chase gated_pointer_chase reachability sudoku4"
BUNDLE="$WIN_ROOT\\bundle"
DICTS="$WIN_ROOT\\phase2\\dictionaries"

note() { local now; now="$(powershell -NoProfile -Command 'Get-Date -Format s')"; echo "${now//$'\r'/} $*" >> "$LOG"; }

# Wait for mains power and at least 6 GB of free commit (a CUDA process commits about 4 GB here).
wait_host() {
  while true; do
    local state
    state="$(powershell -NoProfile -Command '$b = Get-CimInstance Win32_Battery; $o = Get-CimInstance Win32_OperatingSystem; "$(if ($b) { $b.BatteryStatus } else { 2 }) $([int]($o.FreeVirtualMemory / 1024))"')"
    state="${state//$'\r'/}"
    local battery="${state%% *}" commit="${state##* }"
    if [ "$battery" = "2" ] && [ "${commit:-0}" -ge 6000 ]; then return 0; fi
    note "waiting: battery status $battery, free commit ${commit} MB"
    sleep 300
  done
}

lease() {  # lease <size> <python module and arguments...>
  local size="$1"; shift
  "$VRAMORAMA" run --vram "$size" --owner rmp-queue --guard-power 120 --restart 5 -- "$PY" -u -m "$@"
}

# Rerun a resumable job after a failure (a GPU reset or a power cap outlasting vramorama's restarts),
# up to 10 times, 5 minutes apart. Training cells resume exactly and every stage is verified by hash.
retry() {
  local attempt code
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    "$@"; code=$?
    [ "$code" -eq 0 ] && return 0
    note "attempt $attempt ended with exit $code: $*"
    sleep 300
    wait_host
  done
  return "$code"
}

slice() {  # one 20-minute training slice, as the Phase 0 queue ran them; a slice that pauses exits 10
  local arm="$1" seed="$2" variant="$3" steps="$4"; shift 4
  lease 2G research_gym.scripts.run_rmp_training_cell --arm "$arm" --seed "$seed" --families $FAMILIES \
    --variant "$variant" --max-steps "$steps" --batch-per-family 256 --microbatch 256 --learning-rate 0.001 \
    --warmup-steps 200 --eval-every 250 --beta 1.0 --curriculum --staircase --stop-at-gate \
    --result-dir "$WIN_ROOT\\rebuild\\runs" --pool-dir "$WIN_ROOT\\pool" --artifact-dir "$WIN_ROOT\\artifacts" \
    --device cuda --vram-mb 2500 --threads 10 --max-seconds 1200 "$@"
}

cell() {  # cell <arm> <seed> <variant> <max steps> [extra cell arguments...]
  local code failures=0
  while true; do
    slice "$@"; code=$?
    if [ "$code" -eq 0 ]; then return 0; fi
    if [ "$code" -ne 10 ]; then
      failures=$((failures + 1))
      note "slice ended with exit $code ($failures failures in a row): cell $*"
      [ "$failures" -ge 10 ] && return "$code"
      sleep 300
    else
      failures=0
    fi
    wait_host
  done
}

snapshot() {  # snapshot <cell>: the next stage's warm start, as the Phase 0 queue made it
  local source="$ROOT/artifacts/$1/checkpoints/latest.pt" target="$ROOT/snapshots/$1-final.pt"
  mkdir -p "$ROOT/snapshots"
  cp "$source" "$target" && cp "$source.receipt.json" "$target.receipt.json" || return 1
  local expected
  expected="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" "$(cygpath -w "$target.receipt.json")")"
  [ "$(sha256sum "$target" | cut -d' ' -f1)" = "$expected" ]
}

verify() {  # verify <cell>...: every checkpoint must match the committed run record
  "$PY" -u -m research_gym.scripts.rmp_verify_rebuild --cells "$@" --rebuilt-runs "$WIN_ROOT\\rebuild\\runs" \
    --artifact-dir "$WIN_ROOT\\artifacts"
}

phase2() {  # phase2 <lease size> <module suffix> [arguments...]: a Phase 2 script with the work-root paths
  local size="$1" script="$2"; shift 2
  retry lease "$size" "research_gym.scripts.rmp_phase2_$script" --artifact-dir "$WIN_ROOT\\artifacts" "$@"
}
