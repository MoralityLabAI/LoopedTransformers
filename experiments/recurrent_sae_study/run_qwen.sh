#!/usr/bin/env bash
# Qwen3 path for one recovered-student run of the 'Recurrent recovery' sibling session.
#   usage: ./run_qwen.sh <sib_root> <run dir relative to sib_root> <tag> [device] [scale]
#   e.g.   ./run_qwen.sh "$SIB" runs/main_D q06_main cuda 1.0
# Re-snapshot the sibling's looprec into vendor/looprec2 first if its code changed since the run was trained.
set -euo pipefail
ROOT="$1"; RUN="$2"; TAG="$3"; DEV="${4:-cpu}"; SCALE="${5:-1.0}"
PY=/c/Users/patri/Documents/GitHub/MoralityLabAI/Adict/.venv/Scripts/python.exe
cd "$(dirname "$0")"
$PY -u scripts/q1_collect.py --sib_root "$ROOT" --run "$RUN" --tag "$TAG" --device "$DEV" --scale "$SCALE"
T=$($PY -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['T'])" "/c/Users/patri/.cache/recsae_data/mini/$TAG/meta.json")
TM=$($PY -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['Tmax'])" "/c/Users/patri/.cache/recsae_data/mini/$TAG/meta.json")
NAME="mini:${TAG}:rec"
$PY -u scripts/s1_train_sae.py --name "$NAME" --K_train "$T" --device "$DEV"
$PY -u scripts/m2_compare.py --tag "$TAG" --seed 0
$PY -u scripts/t3_toy_analyze.py --name "$NAME" --H 2 --K_train "$T"
$PY -u scripts/t5_toy_controller.py --name "$NAME" --K_train "$T" --cmax "$TM"
