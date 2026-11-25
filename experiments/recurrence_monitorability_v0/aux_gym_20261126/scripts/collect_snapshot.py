"""Snapshot every aux-gym cell's receipts into the push worktree and print a status table (markdown)."""
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

RUNS = Path(r"C:\Users\patri\rmp-aux-20261125\experiments\recurrence_monitorability_v0\runs")
DEST = Path(sys.argv[1])
cells = sorted(p for p in RUNS.iterdir() if p.is_dir() and any(k in p.name for k in ("aux2cpu", "mask2", "modnc")))
rows = []
for c in cells:
    out = DEST / "cells" / c.name
    out.mkdir(parents=True, exist_ok=True)
    for f in ("events.jsonl", "result.json"):
        if (c / f).exists():
            shutil.copyfile(c / f, out / f)
    ev = [json.loads(l) for l in (c / "events.jsonl").read_text().splitlines() if l.strip()]
    evals = [e for e in ev if e.get("event") == "eval"]
    last = evals[-1] if evals else None
    unlocks = [(e["family"], e["depth"], e["step"]) for e in ev if e.get("event") == "unlock"]
    res = json.loads((c / "result.json").read_text()) if (c / "result.json").exists() else None
    dev = next((e["device"] for e in ev if e.get("event") in ("start", "resume")), "?")
    fam = {}
    if last:
        for f, m in last["validation"].items():
            fam[f] = (round(m["exact"], 4), m["by_depth"])
    rows.append({"cell": c.name, "device": dev, "step": last["step"] if last else 0, "status": (res or {}).get("status", "running"),
                 "gate_passed": (res or {}).get("gate_passed"), "families": fam,
                 "max_unlocked": {f: max([d for ff, d, s in unlocks if ff == f], default=None) for f in fam}})
(DEST / "cell_status.json").write_text(json.dumps({"snapshot_utc": datetime.now(timezone.utc).isoformat(), "cells": rows}, indent=1))
print("| Cell | Device | Step | Status | Gate | Interim exact (validation subset, 512) | Deepest unlocked depth |")
print("|---|---|---:|---|---|---|---|")
for r in rows:
    ex = "; ".join(f"{f} {v[0]:.3f}" for f, v in r["families"].items())
    dp = "; ".join(f"{f}: {d}" for f, d in r["max_unlocked"].items())
    print(f"| `{r['cell']}` | {r['device']} | {r['step']:,} | {r['status']} | {r['gate_passed']} | {ex} | {dp} |")
