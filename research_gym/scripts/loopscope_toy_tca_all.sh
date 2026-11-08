#!/usr/bin/env bash
# LOOPSCOPE-01 Q4 TCA (amendments 3-4): CPU, every finished toy arm; waits for the GPU sweep's report before the last arms.
set -u
cd "$(dirname "$0")/../.."
PY="${PY:-/c/Users/patri/.venvs/bluebeam/Scripts/python.exe}"
S=research_gym/scripts/loopscope_toy_tca.py
TOY=experiments/recurrence_monitorability_v0/loopscope_toy
pass() {
  for seed in 1 2 3; do
    for lam in 0 0.01 0.1 1; do
      arm="lam${lam}_s${seed}"
      [ -f "$TOY/ckpt/$arm.final.pt" ] && [ -f "$TOY/runs/$arm/metrics.json" ] || continue
      "$PY" -u "$S" run --arm "$arm" || echo "run $arm exit $?"
      "$PY" -u "$S" control --arm "$arm" || echo "control $arm exit $?"
    done
  done
}
pass
until [ -f "$TOY/report.md" ]; do sleep 120; done
pass
"$PY" -u "$S" report
echo "tca_all finished"
