#!/usr/bin/env bash
# LOOPSCOPE-01 Q4: train + eval all 12 runs sequentially (skips finished), then report.
# Exit code 75 from a train step (power cap) stops the launcher; rerun to resume.
set -u
cd "$(dirname "$0")/../.."
PY="${PY:-/c/Users/patri/.venvs/bluebeam/Scripts/python.exe}"
OUT="${OUT:-experiments/recurrence_monitorability_v0/loopscope_toy}"
DEVICE="${DEVICE:-cuda}"
for seed in ${SEEDS:-1 2 3}; do
  for lam in ${LAMS:-0 0.01 0.1 1}; do
    "$PY" -u research_gym/scripts/loopscope_toy.py train --lam "$lam" --seed "$seed" --out "$OUT" --device "$DEVICE" "$@"
    rc=$?
    if [ $rc -ne 0 ]; then echo "train lam=$lam seed=$seed exit $rc"; exit $rc; fi
    "$PY" -u research_gym/scripts/loopscope_toy.py eval --lam "$lam" --seed "$seed" --out "$OUT" --device "$DEVICE" || exit $?
  done
done
[ -n "${NO_REPORT:-}" ] || "$PY" -u research_gym/scripts/loopscope_toy.py report --out "$OUT"
