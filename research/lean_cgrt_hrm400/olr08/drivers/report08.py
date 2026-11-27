"""Summarise an OLR-08A run directory into report.json, standard library only.

Reads the stage checkpoints exactly as written by run08.py (verifying each .sha256 seal),
recomputes the registered gate quantities, classifies every unsolved search as censored by
a cap (steps, RPCs, heartbeats) or as an exhausted finite action space, totals the metered
resources per stage (native RPCs by op, raw heartbeats, server ms, wall seconds, opening and
replay), and repeats the headline counts without the roots observed by the smoke run.

Usage: python report08.py <run dir> <out json> [smoke-observed root id ...]
"""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

run, out = Path(sys.argv[1]), Path(sys.argv[2])
smoke_ids = set(sys.argv[3:])


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_stage(name):
    d = run / name
    if not d.is_dir():
        return None
    rows = []
    for p in sorted(d.glob("*.json")):
        seal = p.with_suffix(".sha256")
        if not seal.exists() or seal.read_text().strip() != sha(p):
            raise ValueError("unsealed or changed checkpoint " + str(p))
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


reg = json.loads((run / "registry.json").read_text(encoding="utf-8"))
assert sha(run / "registry.json") == (run / "registry.sha256").read_text().strip()
by_id = {r["id"]: r for r in reg}


def validity(rows):
    sc = [r for r in rows if r["scored"]]
    hum = [r["human"] for r in sc if r["human"] is not None]
    return {"selected": len(rows), "scored": len(sc), "T_human": len(hum),
            "human_closed": sum(h["closed"] for h in hum),
            "human_rate": sum(h["closed"] for h in hum) / max(1, len(hum)),
            "menu_any": sum(r["any"] for r in sc), "hard": sum(not r["any"] for r in sc),
            "exclusions": dict(Counter(r["reason"] for r in rows if not r["scored"]))}


def menu_actions(rows):
    sc = [r for r in rows if r["scored"]]
    acts = [m["tactic"] for m in sc[0]["menu"]] if sc else []
    return {a: sum(r["menu"][j]["closed"] for r in sc) for j, a in enumerate(acts)}


def fate(r):
    if r["solved"]:
        return "solved"
    m = r["meter"]
    lim = m["limits"]
    if m["steps"] >= lim[0]:
        return "censored_steps"
    if m["rpcs"] >= lim[1]:
        return "censored_rpcs"
    if m["hb_raw"] >= lim[2]:
        return "censored_hb"
    return "exhausted_action_space"


def resources(rows):
    ops = Counter()
    tot = Counter()
    for r in rows:
        m = r["meter"]
        ops.update(m["ops"])
        tot["steps"] += m["steps"]
        tot["rpcs"] += m["rpcs"]
        tot["hb_raw"] += m["hb_raw"]
        tot["server_ms"] += m["server_ms"]
        tot["native_seconds"] += m["native_seconds"]
        tot["feature_seconds"] += m.get("feature_seconds", 0)
        tot["inference_seconds"] += m.get("inference_seconds", 0)
        tot["search_elapsed_seconds"] += r.get("elapsed", 0)
        o = r.get("open") or {}
        tot["open_seconds"] += o.get("wall_seconds") or 0
        tot["open_hb_raw"] += o.get("hb_raw") or 0
        rp = r.get("replay")
        if rp:
            tot["replays"] += 1
            tot["replay_open_seconds"] += rp["open"].get("wall_seconds") or 0
            tot["replay_native_seconds"] += rp["meter"]["native_seconds"]
            tot["replay_hb_raw"] += rp["meter"]["hb_raw"]
            tot["replay_rpcs"] += rp["meter"]["rpcs"]
    return {"ops": dict(ops), **dict(tot)}


def arm(rows):
    fates = Counter(fate(r) for r in rows)
    solved = [r for r in rows if r["solved"]]
    return {"roots": len(rows), "solved": len(solved), "rate": len(solved) / max(1, len(rows)),
            "fates": dict(fates), "replayed": sum("replay" in r for r in solved),
            "solved_steps": sorted(r["steps"] for r in solved),
            "path_lengths": dict(Counter(len(r["path"]) for r in solved)),
            "by_file": {f: [sum(x["solved"] for x in rows if x["file"] == f), sum(x["file"] == f for x in rows)]
                        for f in sorted({x["file"] for x in rows})},
            "resources": resources(rows)}


def paired(a, b, seed=20270129):
    ai = {r["id"]: r for r in a}
    bi = {r["id"]: r for r in b}
    if ai.keys() != bi.keys():
        return None
    diffs = defaultdict(list)
    for k, r in ai.items():
        diffs[r["file"]].append(int(r["solved"]) - int(bi[k]["solved"]))
    fs = sorted(diffs)
    if not fs:
        return None
    rng = random.Random(seed)
    v = []
    for _ in range(2000):
        x = [d for f in rng.choices(fs, k=len(fs)) for d in diffs[f]]
        v.append(sum(x) / len(x))
    v.sort()
    n = len(ai)
    return {"a_minus_b": (sum(r["solved"] for r in a) - sum(r["solved"] for r in b)) / n,
            "ci95_file_bootstrap": [v[50], v[1949]], "files": len(fs), "roots": n,
            "a_only": sum(ai[k]["solved"] and not bi[k]["solved"] for k in ai),
            "b_only": sum(bi[k]["solved"] and not ai[k]["solved"] for k in ai)}


rep = {"schema": "olr08a.report.v1", "run": str(run), "registry_sha256": sha(run / "registry.json"),
       "contract": json.loads((run / "contract.json").read_text(encoding="utf-8")),
       "smoke_observed_ids": sorted(smoke_ids), "stages": {}}
for f in ("summary.json", "frozen_policy.json", "prior.json", "resource.json"):
    if (run / f).exists():
        rep[f.replace(".json", "") + "_sha256"] = sha(run / f)
        if f != "prior.json":
            rep[f.replace(".json", "")] = json.loads((run / f).read_text(encoding="utf-8"))

ms = load_stage("menu_source")
if ms is not None:
    for role in ("train", "calibration"):
        rows = [r for r in ms if by_id[r["id"]]["role"] == role]
        rep["stages"]["menu_source_" + role] = {"validity": validity(rows), "menu_actions": menu_actions(rows)}
        rows2 = [r for r in rows if r["id"] not in smoke_ids]
        rep["stages"]["menu_source_" + role]["validity_without_smoke_roots"] = validity(rows2)
    rep["stages"]["menu_source_resources"] = {
        "open_seconds": sum((r.get("open") or {}).get("wall_seconds") or 0 for r in ms),
        "replay_open_seconds": sum(((r.get("replay") or {}).get("open") or {}).get("wall_seconds") or 0 for r in ms),
        "native_seconds": sum((r.get("meter") or {}).get("native_seconds", 0) for r in ms),
        "hb_raw": sum((r.get("meter") or {}).get("hb_raw", 0) for r in ms)}

stage_names = ["source_s2", "cal_s1", "cal_s2", "cal_s3", "cal_s3c", "cal_s3_long",
               "eval_s1", "eval_s2", "eval_s3", "eval_s3c"]
stage_names += sorted(p.name for p in run.iterdir() if p.is_dir() and p.name.startswith(("cal_model_", "eval_model_")))
for name in stage_names:
    rows = load_stage(name)
    if rows is None:
        continue
    rep["stages"][name] = arm(rows)
    if smoke_ids & {r["id"] for r in rows}:
        rep["stages"][name]["without_smoke_roots"] = {
            k: v for k, v in arm([r for r in rows if r["id"] not in smoke_ids]).items() if k in ("roots", "solved", "rate", "fates")}

cal = {a: load_stage("cal_" + a) for a in ("s1", "s2", "s3", "s3c")}
if all(cal.values()):
    long = load_stage("cal_s3_long")
    best_arm = max(cal, key=lambda a: sum(r["solved"] for r in cal[a]))
    rep["paired_calibration"] = {f"{a}_vs_{b}": paired(cal[a], cal[b]) for a in cal for b in cal if a < b}
    if long:
        rep["paired_calibration"]["s3_long_vs_best"] = paired(long, cal[best_arm])
        rep["paired_calibration"]["best_same_budget_arm"] = best_arm

mf = load_stage("menu_fresh")
if mf is not None:
    rep["stages"]["menu_fresh"] = {"validity": validity(mf), "menu_actions": menu_actions(mf),
                                   "validity_without_smoke_roots": validity([r for r in mf if r["id"] not in smoke_ids])}

out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print("REPORT", out, {k: (v.get("solved"), v.get("roots")) if "solved" in v else "menu" for k, v in rep["stages"].items()})
