#!/usr/bin/env bash
set -euo pipefail
ROOT="$(realpath "${1:?workspace}")"
ROSTER="$(realpath "${2:?roster}")"
N="${3:?root count}"
TOKENS="${4:?max new tokens}"
SECONDS_LIMIT="${5:?capture seconds}"
SHARD="${6:?shard id}"
[[ "$SHARD" =~ ^[a-zA-Z0-9_-]+$ ]]
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/venv/bin/python"
OUT="$ROOT/results/$SHARD"
mkdir -p "$ROOT/results"
mkdir "$OUT"
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
timeout --kill-after=30 "$((SECONDS_LIMIT+90))" "$PY" "$HERE/gym_online.py" --snapshot "$ROOT/source_snapshot" --runtime-receipt "$ROOT/runtime_receipt.json" --roster "$ROSTER" --out "$OUT/raw" --expected-roots "$N" --max-new-tokens "$TOKENS" --limit-seconds "$SECONDS_LIMIT"
"$PY" - "$OUT/raw/online_receipt.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="complete" and all(v["complete"] for v in r["arms"].values())
PY
"$PY" "$HERE/analyze.py" --pilot-root "$OUT/raw" --out "$OUT/analysis"
cp "$OUT/raw/online_receipt.json" "$OUT/raw/online_prefix_check.json" "$ROOT/inventory.json" "$ROOT/runtime_receipt.json" "$OUT/analysis/"
"$PY" "$HERE/gym_manifest.py" "$OUT/analysis" --exporter-commit "$(git -C "$HERE" rev-parse HEAD)"
"$PY" "$HERE/gym_manifest.py" "$OUT/analysis" --verify
echo "Verified $OUT/analysis. Transfer it and terminate the rental manually."
