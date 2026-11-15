#!/usr/bin/env bash
# RAM-10 WSL/Linux pinned source-only TRAIN or frozen TEST. Does not merge.
set -euo pipefail

usage() {
  echo 'Usage:' >&2
  echo '  bash experiments/lean_ram10/run_local.sh train MATHCAN_TRAIN GYM LEAN_BIN NEW_OUTPUT_DIR' >&2
  echo '  bash experiments/lean_ram10/run_local.sh test  MATHCAN_TEST  GYM LEAN_BIN NEW_OUTPUT_DIR FROZEN_DATA FROZEN_TRAINING' >&2
  exit 2
}
[[ $# -ge 5 ]] || usage
mode="$1"; mathcan="$(realpath "$2")"; gym="$(realpath "$3")"; leanbin="$(realpath "$4")"; out="$(realpath -m "$5")"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config="$repo/experiments/lean_ram10/config.json"
[[ -f "$config" && -x "$leanbin/lean" && -x "$leanbin/lake" ]] || { echo 'Missing RAM-10 config or Lean4.35 binary directory' >&2; exit 1; }
[[ ! -e "$out" ]] || { echo "Output already exists: $out. Preserve every first-run receipt." >&2; exit 1; }
[[ "$(git -C "$mathcan" rev-parse HEAD)" == 7515eecf51a60e7f8cfcd0801fddf8a3cf039e89 ]] || { echo 'MathCan SHA mismatch' >&2; exit 1; }
[[ "$(git -C "$gym" rev-parse HEAD)" == e8a780254b5cf41d479095f4f22bcfc4aa76ed72 ]] || { echo 'Original compiler gym SHA mismatch' >&2; exit 1; }
[[ "$(cat "$mathcan/lean-toolchain")" == 'leanprover/lean4:v4.35.0' ]] || { echo 'Wrong pinned Lean toolchain' >&2; exit 1; }
[[ "$(git -C "$repo" hash-object experiments/lean_ram10/config.json)" == 4584555cbae1f7c0e95a29b6df42c85a62ac2a10 ]] || { echo 'Original registered RAM-10 config blob changed' >&2; exit 1; }
export PATH="$leanbin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LEAN_NUM_THREADS=2 PYTHONHASHSEED=0
export PYTHONPATH="$repo/experiments/lean_ram10:$gym${PYTHONPATH:+:$PYTHONPATH}"

build_server() {
  export LEAN_PATH="$repo/experiments/lean_ram03:$(cd "$mathcan" && lake env printenv LEAN_PATH)"
  lean "$repo/experiments/lean_ram03/ProbeServer.lean" -o "$repo/experiments/lean_ram03/ProbeServer.olean"
}

case "$mode" in
  train)
    [[ $# -eq 5 ]] || usage
    for name in NOR07Volterra PeriodOne TacticFibers TreeReconstruction; do
      [[ ! -e "$mathcan/MathCan/$name.lean" ]] || { echo "TRAIN checkout still contains heldout $name. Make a separate disposable masked checkout." >&2; exit 1; }
    done
    python - "$mathcan" "$config" train <<'PYRAM10CHECK'
import json,sys
from pathlib import Path
from ram10_core import verify_checkout
print('RAM10_PINNED_MASKED_SOURCE_VERIFIED',verify_checkout(Path(sys.argv[1]),json.loads(Path(sys.argv[2]).read_text()),sys.argv[3]))
PYRAM10CHECK
    (cd "$mathcan" && lake build MathCan.BasinCapture MathCan.BlockRemainder MathCan.CoarseStateDependence MathCan.FilteredOrbit MathCan.InvariantBall MathCan.MemoryProjection MathCan.MultiBasin MathCan.OrbitStitching MathCan.GradientStability MathCan.OrderedJet)
    build_server
    (cd "$repo" && python -m unittest discover -s experiments/lean_ram10 -p 'test_*.py' -v)
    mkdir -p "$out"
    python "$repo/experiments/lean_ram10/collect.py" --mathcan "$mathcan" --config "$config" --lean "$leanbin/lean" --out "$out/data"
    python "$repo/experiments/lean_ram10/train.py" --source "$out/data" --config "$config" --out "$out/training"
    echo "RAM-10 original nine-fit TRAIN artifact: $out"
    ;;
  test)
    [[ $# -eq 7 ]] || usage
    data="$(realpath "$6")"; frozen="$(realpath "$7")"
    # This verifies byte-identical original training source/model receipts BEFORE
    # the test source file contents or compiler goal states are opened.
    python - "$config" "$data" "$frozen" <<'PY'
from pathlib import Path
import sys
from evaluate import verify_frozen
_,rec,checkpoint=verify_frozen(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
print('RAM10_SOURCE_ONLY_FROZEN_BEFORE_TEST',sorted(checkpoint),rec['dataset_sha256'])
PY
    python - "$mathcan" "$config" test <<'PYRAM10CHECK'
import json,sys
from pathlib import Path
from ram10_core import verify_checkout
print('RAM10_PINNED_PRISTINE_SOURCE_VERIFIED',verify_checkout(Path(sys.argv[1]),json.loads(Path(sys.argv[2]).read_text()),sys.argv[3]))
PYRAM10CHECK
    (cd "$mathcan" && lake build MathCan.NOR07Volterra MathCan.PeriodOne MathCan.TacticFibers MathCan.TreeReconstruction)
    build_server
    python "$repo/experiments/lean_ram10/evaluate.py" --mathcan "$mathcan" --data "$data" --training "$frozen" --config "$config" --lean "$leanbin/lean" --out "$out/results"
    python "$repo/experiments/lean_ram10/audit_results.py" --results "$out/results" --config "$config" --lean "$leanbin/lean" --out "$out/third_replay"
    echo "RAM-10 original theorem and third Lean certificate audits: $out"
    ;;
  *) usage ;;
esac
