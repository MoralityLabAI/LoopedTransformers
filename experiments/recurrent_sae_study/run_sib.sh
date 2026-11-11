#!/usr/bin/env bash
# Reduced toy-path pipeline over crrbench sibling checkpoints.  usage: ./run_sib.sh <substring-or-ALL> [threads] [stages]
#   stages (default): collect,sae,t3,t4b,t7,t8   optional extra: t5,t6
set -uo pipefail
SUB="${1:-ALL}"; TH="${2:-4}"; STAGES="${3:-collect,sae,t3,t4b,t7,t8}"
PY=/c/Users/patri/Documents/GitHub/MoralityLabAI/Adict/.venv/Scripts/python.exe
cd "$(dirname "$0")"
[ "$SUB" = "ALL" ] && SUB=""
for NAME in $($PY scripts/sib_models.py list "$SUB" | tr -d '\r'); do
  B=$(basename "${NAME#sib:}" .pt)
  LOG=out/logs/sib_${B}.log
  echo "=== $B  $(date +%H:%M:%S)" | tee -a "$LOG"
  has() { [[ ",$STAGES," == *",$1,"* ]]; }
  KT=$($PY -c "import sys; sys.path.insert(0,'src'); import recsae; from recsae import toy; print(toy.load_toy(sys.argv[1])[2]['cfg']['K'])" "$NAME")
  has collect && $PY -u scripts/t2_toy_collect_sae.py --name "$NAME" --threads "$TH" >> "$LOG" 2>&1
  has sae     && { if [ "${USE_GPU:-0}" = "1" ]; then /c/Users/patri/AppData/Local/Programs/vramorama/vramorama.exe run --vram 1G --owner recsae-sae -- $PY -u scripts/s1_train_sae.py --name "$NAME" --K_train "$KT" --steps 2000 --device cuda >> "$LOG" 2>&1; else $PY -u scripts/s1_train_sae.py --name "$NAME" --K_train "$KT" --steps 2000 --threads "$TH" >> "$LOG" 2>&1; fi; }
  has t3      && $PY -u scripts/t3_toy_analyze.py --name "$NAME" --tag "$B" --H 4 --threads "$TH" --boot 100 >> "$LOG" 2>&1
  has t4b     && $PY -u scripts/t4b_positive_control.py --name "$NAME" --tag "$B" --t0s 2,4,8,16,24 --ms 2,8 --n_eval 400 --threads "$TH" >> "$LOG" 2>&1
  has t7      && $PY -u scripts/t7_loss_attribution.py --name "$NAME" --tag "$B" --threads "$TH" >> "$LOG" 2>&1
  has t8      && $PY -u scripts/t8_reuse.py --name "$NAME" --tag "$B" --threads "$TH" --boot 100 --delays 0,4 >> "$LOG" 2>&1
  has t5      && $PY -u scripts/t5_toy_controller.py --name "$NAME" --tag "$B" --threads "$TH" --boot 150 >> "$LOG" 2>&1
  has t6      && $PY -u scripts/t6_hrm_sae.py --name "$NAME" --tag "$B" --threads "$TH" --seeds 2 --iters 200 --boot 100 >> "$LOG" 2>&1
  echo "--- done $B $(date +%H:%M:%S)" | tee -a "$LOG"
done
