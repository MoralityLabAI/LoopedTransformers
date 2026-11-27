#!/usr/bin/env bash
# Relaunch the registered run on the same --out after an external kill (exit without a
# Python traceback); stop on a traceback, on success, or after MAX attempts. Every attempt
# has its own log and a ledger line with start/end UTC times and the exit code.
export PATH="/c/Users/patri/.elan/toolchains/leanprover--lean4---v4.19.0/bin:/c/Users/patri/AppData/Local/GitHubDesktop/app-3.6.6/resources/app/git/cmd:$PATH"
cd /c/olr08 || exit 2
OUT=C:/olr08/full2
LEDGER=C:/olr08/full2_attempts.tsv
MAX=25
n=$(( $(ls C:/olr08/full2.attempt*.log 2>/dev/null | wc -l) + 1 ))
while [ "$n" -le "$MAX" ]; do
  log=C:/olr08/full2.attempt$n.log
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  /c/olr08/venv/Scripts/python.exe run_full.py C:/olr08/pkg2 --source D:/lean4-419-source \
    --exe C:/olr08/pkg/lean/.lake/build/bin/lgym.exe --out "$OUT" --profile full --threads 8 --workers 2 \
    --ram-gib 12 --reissue-a C:/olr08/reissues/scope-reissue-07a/manifest.json \
    --reissue-b C:/olr08/reissues/scope-reissue-07b/manifest.json > "$log" 2>&1
  code=$?
  printf '%s\t%s\t%s\t%s\n' "$n" "$start" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code" >> "$LEDGER"
  if [ "$code" -eq 0 ]; then echo "RUN_DONE attempt $n"; exit 0; fi
  if grep -q "Traceback" "$log"; then echo "RUN_ERROR attempt $n exit $code"; tail -5 "$log"; exit 1; fi
  echo "RUN_KILLED attempt $n exit $code, resuming"
  n=$((n + 1))
  sleep 20
done
echo "RUN_GAVE_UP after $MAX attempts"
exit 3
