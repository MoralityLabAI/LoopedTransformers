#!/usr/bin/env bash
# LOOPSCOPE-01 Q5 (amendment 5): 12 fresh toy runs, TCA F1/F8 per run, then the Q5 decision. Resumable.
set -u
cd "$(dirname "$0")/../.."
PY="${PY:-/c/Users/patri/.venvs/bluebeam/Scripts/python.exe}"
TOY=experiments/recurrence_monitorability_v0/loopscope_toy_q5
TCA=experiments/recurrence_monitorability_v0/loopscope_toy_q5_tca
SEEDS="4 5 6 7 8 9" LAMS="0 0.01" OUT="$TOY" NO_REPORT=1 DEVICE=cuda \
  bash research_gym/scripts/loopscope_toy_all.sh --fixed-depth --loops 8 --curriculum-frac 0 --lr 3e-4 --steps 30000 || exit $?
for seed in 4 5 6 7 8 9; do
  for lam in 0 0.01; do
    "$PY" -u research_gym/scripts/loopscope_toy_tca.py run --arm "lam${lam}_s${seed}" --device cuda --toy "$TOY" --out "$TCA" || exit $?
  done
done
"$PY" -u research_gym/scripts/loopscope_toy_tca.py q5 --toy "$TOY" --out "$TCA"
echo "q5 finished"
