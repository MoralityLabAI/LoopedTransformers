#!/usr/bin/env bash
# CPU smoke test of every Phase 2 code path on a real model (the random-initialization LOOP-T s1 C0),
# with short training and few permutations, writing only to <out-dir>. Prints SMOKE OK at the end.
# Usage: scripts/smoke_rmp_phase2.sh <out-dir> [work-root]   (run from the repository root)
set -u
OUT="$(cygpath -w "${1:?out dir}")"
R="$(cygpath -w "${2:-/c/Users/patri/Documents/Codex/rmp_v0}")"
PY="${PY:-/c/Users/patri/.venvs/bluebeam/Scripts/python.exe}"
M="LOOP-T s1 C0"
common=(--artifact-dir "$R\artifacts" --monitor-bundle "$R\bundle" --metrics "$OUT\dictionaries.json" --device cpu)
dict() { "$PY" -u -m research_gym.scripts.rmp_phase2_dictionaries --model "$M" "${common[@]}" --work-dir "$OUT\dicts" \
  --steps 50 --eval-examples 200 "$@" || { echo "FAILED dictionaries $*"; exit 1; }; }
step() { echo "== $*"; "$PY" -u -m "$@" || { echo "FAILED $1"; exit 1; }; }
dict --kind shared --widths 1024 --ks 8
dict --kind shared --seeds 1 2 --selected-from "$M"
dict --kind tok --selected-from "$M"
dict --kind visit --sites 1 2 3 4 5 6 7 8 --selected-from "$M"
dict --kind gauss --selected-from "$M"
dict --kind centered --selected-from "$M"
dict --kind trans --widths 1024 --ks 8
dict --kind trans --selected-from "$M" --selected-kind trans --adjacent
step research_gym.scripts.rmp_phase2_association --models "$M" --rounds 5 --out "$OUT\association.json" "${common[@]}"
step research_gym.scripts.rmp_phase2_association --kind gauss --models "$M" --rounds 5 --out "$OUT\association.json" "${common[@]}"
step research_gym.scripts.rmp_phase2_reported --models "$M" --association "$OUT\association.json" --out "$OUT\reported.json" "${common[@]}"
step research_gym.scripts.rmp_phase2_controls --models "$M" --out "$OUT\controls.json" "${common[@]}"
step research_gym.scripts.rmp_phase2_n1 --model "$M" --out "$OUT\n1.json" "${common[@]}"
echo SMOKE OK
