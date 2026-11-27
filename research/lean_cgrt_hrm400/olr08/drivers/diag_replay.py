"""Diagnostic only: rerun the registered S2 search on given roots and, if the fresh replay
of the credited path fails, record each replay step's native response. Writes nothing into
the run directory."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/olr08/pkg")
import run08  # noqa: E402

EXE = Path("C:/olr08/pkg/lean/.lake/build/bin/lgym.exe")
SRC = Path("D:/lean4-419-source")
rows = json.load(open(sys.argv[1], encoding="utf-8"))[: int(sys.argv[2])]
out = Path(sys.argv[3])
report = []
orig = run08.replay


def traced_replay(exe, source, row, path, prior=None, frames=False):
    try:
        return orig(exe, source, row, path, prior, frames)
    except ValueError as e:
        steps = []
        with run08.Lean(exe) as L:
            s, why, _ = run08.open_root(L, source, row)
            for tac in path:
                r = L.request(op="step", sid=s["sid"], tactic=tac, hb=500 if tac == "exact?" else 200)
                steps.append({"tactic": tac, "ok": r.get("ok"), "ngoals": r.get("ngoals"), "error": r.get("error"),
                              "hb": r.get("hb"), "request_hb_raw": r.get("request_hb_raw"), "ms": r.get("ms")})
                if r.get("ok") is not True:
                    break
                s = r
        raise RuntimeError(json.dumps({"replay_error": str(e), "steps": steps}))


run08.replay = traced_replay
for row in rows:
    try:
        r = run08.search(EXE, SRC, row, "s2")
        rec = {"id": row["id"], "decl": row["decl"], "solved": r["solved"], "path": r["path"], "replay_ok": True,
               "path_events": [r["events"][i] for i in r["on_path"]] if r["solved"] else []}
    except RuntimeError as e:
        rec = {"id": row["id"], "decl": row["decl"], "replay_ok": False, **json.loads(str(e))}
    report.append(rec)
    print(json.dumps(rec)[:1500], flush=True)
out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
