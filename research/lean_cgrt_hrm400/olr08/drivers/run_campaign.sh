#!/usr/bin/env bash
# Usage: run_campaign.sh <package dir> <out dir> <expected registry sha256>
# 1. Preflight: freeze the registry with the unchanged full-profile prelude, stop before outcomes.
# 2. Refuse to continue unless the registry equals the audited one.
# 3. Run the campaign; after an external kill (exit without a Python traceback) resume the same
#    --out under the same contract; stop on a traceback, success, or MAX attempts.
PKG=$1; OUT=$2; EXPECT=$3; MAX=25
export PATH="/c/Users/patri/.elan/toolchains/leanprover--lean4---v4.19.0/bin:/c/Users/patri/AppData/Local/GitHubDesktop/app-3.6.6/resources/app/git/cmd:$PATH"
cd /c/olr08 || exit 2
tag=$(basename "$OUT")
ARGS=(--source D:/lean4-419-source --exe C:/olr08/pkg/lean/.lake/build/bin/lgym.exe --out "$OUT" --profile full
      --threads 8 --workers 2 --ram-gib 12 --reissue-a C:/olr08/reissues/scope-reissue-07a/manifest.json
      --reissue-b C:/olr08/reissues/scope-reissue-07b/manifest.json)
LEDGER=C:/olr08/${tag}_attempts.tsv
if [ ! -f "$OUT/registry.sha256" ]; then
  for i in 1 2 3 4 5; do
    /c/olr08/venv/Scripts/python.exe preflight_freeze.py "$PKG" "${ARGS[@]}" > "C:/olr08/${tag}.preflight$i.log" 2>&1
    code=$?
    printf 'preflight%s\t%s\t%s\n' "$i" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code" >> "$LEDGER"
    grep -q PREFLIGHT_STOPPED_BEFORE "C:/olr08/${tag}.preflight$i.log" && break
    grep -q Traceback "C:/olr08/${tag}.preflight$i.log" && { echo "PREFLIGHT_ERROR"; exit 1; }
    sleep 20
  done
fi
got=$(sha256sum "$OUT/registry.json" | cut -c1-64)
[ "$got" = "$EXPECT" ] || { echo "REGISTRY_MISMATCH $got"; exit 1; }
echo "REGISTRY_OK $got"
n=$(( $(ls C:/olr08/${tag}.attempt*.log 2>/dev/null | wc -l) + 1 ))
while [ "$n" -le "$MAX" ]; do
  log=C:/olr08/${tag}.attempt$n.log
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  /c/olr08/venv/Scripts/python.exe run_full.py "$PKG" "${ARGS[@]}" > "$log" 2>&1
  code=$?
  printf '%s\t%s\t%s\t%s\n' "$n" "$start" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code" >> "$LEDGER"
  if [ "$code" -eq 0 ]; then echo "RUN_DONE attempt $n"; exit 0; fi
  if grep -q "Traceback" "$log"; then echo "RUN_ERROR attempt $n exit $code"; tail -n 5 "$log"; exit 1; fi
  echo "RUN_KILLED attempt $n exit $code, resuming"
  n=$((n + 1))
  sleep 20
done
echo "RUN_GAVE_UP after $MAX attempts"
exit 3
