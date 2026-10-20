#!/usr/bin/env bash
# Full toy-path analysis for one run name.   usage: ./run_toy.sh <name> [threads]
#   name = own checkpoint (e.g. dfa_random_exit_K16_s0) or sib:<path to crrbench results .../models/<task>__<regime>__s<seed>.pt>
set -euo pipefail
NAME="$1"; TH="${2:-6}"
PY=/c/Users/patri/Documents/GitHub/MoralityLabAI/Adict/.venv/Scripts/python.exe
cd "$(dirname "$0")"
KT=$($PY -c "import sys; sys.path.insert(0,'src'); import recsae; from recsae import toy; print(toy.load_toy(sys.argv[1])[2]['cfg']['K'])" "$NAME")
$PY -u scripts/t2_toy_collect_sae.py --name "$NAME" --threads "$TH"
$PY -u scripts/s1_train_sae.py --name "$NAME" --K_train "$KT" --threads "$TH"
$PY -u scripts/t3_toy_analyze.py --name "$NAME" --H 4 --threads "$TH"
$PY -u scripts/t4_toy_intervene.py --name "$NAME" --threads "$TH"
$PY -u scripts/t5_toy_controller.py --name "$NAME" --threads "$TH"
$PY -u scripts/t6_hrm_sae.py --name "$NAME" --threads "$TH"
