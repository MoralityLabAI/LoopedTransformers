#!/usr/bin/env bash
# Full mini-LaCo replica for one seed.   usage: ./run_mini.sh <seed> [threads]
set -euo pipefail
SEED="$1"; TH="${2:-6}"
PY=/c/Users/patri/Documents/GitHub/MoralityLabAI/Adict/.venv/Scripts/python.exe
cd "$(dirname "$0")"
NAME="mini:dfa_s${SEED}:rec"
$PY -u scripts/m1_build.py --seed "$SEED" --threads "$TH"
$PY -u scripts/s1_train_sae.py --name "$NAME" --K_train 5 --threads "$TH"
$PY -u scripts/m2_compare.py --seed "$SEED" --threads "$TH"
$PY -u scripts/t3_toy_analyze.py --name "$NAME" --H 2 --K_train 5 --threads "$TH"
$PY -u scripts/m3_mini_intervene.py --seed "$SEED" --threads "$TH"
$PY -u scripts/t5_toy_controller.py --name "$NAME" --K_train 5 --cmax 10 --threads "$TH"
