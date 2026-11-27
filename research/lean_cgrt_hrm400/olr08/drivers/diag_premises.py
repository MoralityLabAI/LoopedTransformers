"""Diagnostic only: rerun the registered S1 calibration search on roots without a checkpoint
and record any premise-enumeration failure with its state, under the unrepaired and the
repaired enumerator. Writes nothing into the run directory."""
import glob
import json
import os
import sys
from pathlib import Path

pkg, run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(pkg))
import run08  # noqa: E402

EXE = Path("C:/olr08/pkg/lean/.lake/build/bin/lgym.exe")
SRC = Path("D:/lean4-419-source")
reg = json.load(open(run / "registry.json", encoding="utf-8"))
menu = {json.load(open(p, encoding="utf-8"))["id"]: json.load(open(p, encoding="utf-8"))
        for p in glob.glob(str(run / "menu_source" / "*.json"))}
cal = [r for r in reg if r["role"] == "calibration" and menu[r["id"]]["scored"] and not menu[r["id"]]["any"]]
done = {os.path.basename(p)[:-5] for p in glob.glob(str(run / "cal_s1" / "*.json"))}
todo = [r for r in cal if r["id"] not in done]
failures = []
orig = run08.enumerate_menu


def logged(m, s):
    try:
        return orig(m, s)
    except ValueError as e:
        failures.append({"goals": s.get("goals"), "error": str(e)[:400]})
        raise


run08.enumerate_menu = logged
res = []
for row in todo:
    try:
        r = run08.search(EXE, SRC, row, "s1")
        timeouts = sum(1 for j in r["meter"]["rpc_journal"] if j["op"] == "premises" and j["ok"] is False)
        res.append({"id": row["id"], "decl": row["decl"], "file": row["file"], "completed": True,
                    "solved": r["solved"], "steps": r["steps"], "premise_timeouts": timeouts})
    except ValueError as e:
        res.append({"id": row["id"], "decl": row["decl"], "file": row["file"], "completed": False, "error": str(e)[:300]})
    print(json.dumps(res[-1]), flush=True)
out.write_text(json.dumps({"package": str(pkg), "roots": res, "failures": failures}, indent=1) + "\n", encoding="utf-8")
