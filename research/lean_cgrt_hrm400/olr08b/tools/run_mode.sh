#!/usr/bin/env bash
# Usage: run_mode.sh <tag> <campaign.py args...>
# Runs the registered OLR-08B campaign.py under the isolated venv. After an external kill
# (nonzero exit without a Python traceback) it resumes the same --out under the same
# contract; it stops on a traceback, on success, or after MAX attempts. Each attempt keeps its
# own log and a copy of resource.json; a ledger records UTC start/end and exit codes.
# It never terminates any other process.
TAG=$1; shift
MAX=25
export PATH="/c/Users/patri/.elan/toolchains/leanprover--lean4---v4.19.0/bin:/c/Users/patri/AppData/Local/GitHubDesktop/app-3.6.6/resources/app/git/cmd:$PATH"
export PYTHONNOUSERSITE=1
cd /c/Users/patri/olr08b-399-wt || exit 2
LOGS=C:/olr08b-399/logs; mkdir -p "$LOGS"
LEDGER=$LOGS/${TAG}_attempts.tsv
OUT=""; prev=""; for a in "$@"; do [ "$prev" = "--out" ] && OUT=$a; prev=$a; done
n=$(( $(ls $LOGS/${TAG}.attempt*.log 2>/dev/null | wc -l) + 1 ))
while [ "$n" -le "$MAX" ]; do
  log=$LOGS/${TAG}.attempt$n.log
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  /c/olr08/venv/Scripts/python.exe research/lean_cgrt_hrm400/olr08b/campaign.py "$@" > "$log" 2>&1
  code=$?
  [ -n "$OUT" ] && [ -f "$OUT/resource.json" ] && cp "$OUT/resource.json" "$LOGS/${TAG}.attempt$n.resource.json"
  printf '%s\t%s\t%s\t%s\n' "$n" "$start" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code" >> "$LEDGER"
  if [ "$code" -eq 0 ]; then echo "RUN_DONE $TAG attempt $n"; exit 0; fi
  if grep -q "Traceback" "$log"; then echo "RUN_ERROR $TAG attempt $n exit $code"; tail -n 6 "$log"; exit 1; fi
  echo "RUN_KILLED $TAG attempt $n exit $code, resuming"
  n=$((n + 1))
  sleep 20
done
echo "RUN_GAVE_UP $TAG after $MAX attempts"
exit 3
