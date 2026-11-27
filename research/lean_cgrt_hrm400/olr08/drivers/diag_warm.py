"""Diagnostic only: replay credited paths in a fresh pre-command process, cold and after one
uncharged exact? warm-up at the root with a 20000-thousand-heartbeat cap. Same per-step caps
and certification as run08.replay. Writes nothing into the run directory."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/olr08/pkg")
import run08  # noqa: E402

EXE = Path("C:/olr08/pkg/lean/.lake/build/bin/lgym.exe")
SRC = Path("D:/lean4-419-source")
run = Path(sys.argv[1])
reg = {r["id"]: r for r in json.load(open(run / "registry.json", encoding="utf-8"))}
cases = [(reg[r["id"]], r["path"]) for r in
         (json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(str(run / "source_s2" / "*.json"))))
         if r["solved"] and "exact?" in r["path"]]
for d in json.load(open(sys.argv[2], encoding="utf-8")):
    cases.insert(0, (reg[d["id"]], [s["tactic"] for s in d["steps"]]))


def attempt(row, path, warm):
    with run08.Lean(EXE) as L:
        s, why, _ = run08.open_root(L, SRC, row)
        w = None
        if warm:
            r = L.request(op="step", sid=s["sid"], tactic="exact?", hb=20000)
            w = {"ok": r.get("ok"), "hb_raw": r.get("request_hb_raw"), "ms": r.get("request_ms")}
        hbs = []
        for tac in path:
            s2 = L.request(op="step", sid=s["sid"], tactic=tac, hb=500 if tac == "exact?" else 200)
            hbs.append(s2.get("request_hb_raw"))
            if s2.get("ok") is not True:
                return {"ok": False, "failed_at": tac, "hb": hbs, "warm": w}
            s = s2
        c = L.request(op="cert", sid=s["sid"]).get("cert", {})
        return {"ok": s.get("ngoals") == 0 and run08.close_cert(c), "hb": hbs, "warm": w}


res = []
for row, path in cases:
    cold = attempt(row, path, False)
    warm = attempt(row, path, True)
    res.append({"id": row["id"], "decl": row["decl"], "path": path, "cold": cold, "warm": warm})
    print(row["decl"], path, "cold", cold["ok"], cold["hb"], "warm", warm["ok"], warm["hb"], warm["warm"], flush=True)
Path(sys.argv[3]).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print("SUMMARY cases", len(res), "cold_ok", sum(r["cold"]["ok"] for r in res), "warm_ok", sum(r["warm"]["ok"] for r in res))
