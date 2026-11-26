"""Summarise an OLR-08B run directory (spent or fresh-calibration) into analysis.json.

Standard library only. Verifies every stage checkpoint against its .sha256 seal, then reports:
closure curves by policy and budget; shared-budget portfolio (RR, UCB) against single arms
and the offline union of single arms; premise-retrieval charges and menu-cache use; depth and
premise-width diagnostics; memory tiers; credited-path replay; paired file-cluster bootstrap
comparisons; and, optionally, identity of the 300-step single-arm searches with OLR-08A.

Usage: python analyze08b.py <run dir> <out json> [08A run dir for identity check]
"""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

run, out = Path(sys.argv[1]), Path(sys.argv[2])
ref08a = Path(sys.argv[3]) if len(sys.argv) > 3 else None
ARMS = ("s1", "s2", "s3", "s3c")
POLICIES = (*ARMS, "rr", "ucb")
BUDGETS = (75, 150, 300, 600, 1200)
TIERS = (0, 64, 256, 1024, 4096)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stage(d):
    rows = []
    for p in sorted(d.glob("*.json")):
        seal = p.with_suffix(".sha256")
        if not seal.exists() or seal.read_text().strip() != sha(p):
            raise ValueError("unsealed or changed checkpoint " + str(p))
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def tag(p, b, m=0, d=3, w=False):
    return f"{p}_b{b}_m{m}_d{d}_w{int(w)}"


def compare(a, b, seed=20270129, draws=2000):
    ai = {r["id"]: r for r in a}
    bi = {r["id"]: r for r in b}
    if ai.keys() != bi.keys() or not ai:
        return None
    diffs = defaultdict(list)
    for k, r in ai.items():
        diffs[r["file"]].append(int(r["solved"]) - int(bi[k]["solved"]))
    fs = sorted(diffs)
    rng = random.Random(seed)
    v = sorted(sum(x) / len(x) for x in
               ([d for f in rng.choices(fs, k=len(fs)) for d in diffs[f]] for _ in range(draws)))
    return {"a_minus_b_roots": sum(r["solved"] for r in a) - sum(r["solved"] for r in b),
            "a_minus_b_rate": (sum(r["solved"] for r in a) - sum(r["solved"] for r in b)) / len(ai),
            "ci95_file_bootstrap": [v[int(.025 * draws)], v[int(.975 * draws) - 1]],
            "a_only": sum(ai[k]["solved"] and not bi[k]["solved"] for k in ai),
            "b_only": sum(bi[k]["solved"] and not ai[k]["solved"] for k in ai),
            "roots": len(ai), "files": len(fs)}


def charges(rows):
    c = Counter()
    for r in rows:
        m = r["meter"]
        c["roots"] += 1
        c["solved"] += bool(r["solved"])
        c["steps"] += m["steps"]
        c["rpcs"] += m["rpcs"]
        c["hb_raw"] += m["hb_raw"]
        c["server_ms"] += m["server_ms"]
        c["native_seconds"] += m["native_seconds"]
        c["elapsed_seconds"] += r.get("elapsed", 0)
        c["open_seconds"] += (r.get("open") or {}).get("wall_seconds") or 0
        for j in m["rpc_journal"]:
            if j["op"] == "premises":
                c["premise_rpcs"] += 1
                c["premise_hb_raw"] += j["hb"]
                c["premise_server_ms"] += j["server_ms"]
                c["premise_wall_seconds"] += j["wall_seconds"]
                c["premise_timeouts"] += j["ok"] is False
            elif j["op"] == "cert":
                c["cert_rpcs"] += 1
                c["cert_hb_raw"] += j["hb"]
            elif j["op"] == "step":
                c["step_hb_raw"] += j["hb"]
        ca = r["cache"]
        c["cache_hits"] += ca["hits"]
        c["cache_misses"] += ca["misses"]
        c["cache_evictions"] += ca["evictions"]
        c["cache_seconds"] += ca["seconds"]
        c["cache_peak_payload_max"] = max(c["cache_peak_payload_max"], ca["peak"])
        rp = r.get("replay")
        if rp:
            c["replays"] += 1
            c["replay_hb_raw"] += rp["meter"]["hb_raw"]
            c["replay_open_seconds"] += (rp.get("open") or {}).get("wall_seconds") or 0
            c["replay_warmups"] += len(rp.get("library_search_warmup") or [])
    out = dict(c)
    out["rate"] = c["solved"] / max(1, c["roots"])
    out["premise_hb_share"] = c["premise_hb_raw"] / max(1, c["hb_raw"])
    return out


def fates(rows):
    f = Counter()
    for r in rows:
        if r["solved"]:
            f["solved"] += 1
            continue
        m = r["meter"]
        lim = m["limits"]
        f["censored_steps" if m["steps"] >= lim[0] else "censored_rpcs" if m["rpcs"] >= lim[1]
          else "censored_hb" if m["hb_raw"] >= lim[2] else "exhausted_action_space"] += 1
    return dict(f)


contract = json.loads((run / "contract.json").read_text(encoding="utf-8"))
specs = [tuple(s) for s in contract["specs"]]
res = {}
for s in specs:
    d = run / tag(s[0], s[1], s[2], s[3], bool(s[4]))
    if d.is_dir():
        res[d.name] = stage(d)
menu = stage(run / "menu_cal") if (run / "menu_cal").is_dir() else []
reg = json.loads((run / "registry.json").read_text(encoding="utf-8"))
rep = {"schema": "olr08b.analysis.v1", "run": str(run), "mode": contract["mode"],
       "registry_sha256": sha(run / "registry.json"), "contract_sha256": sha(run / "contract.json"),
       "specs_registered": len(specs), "specs_complete": len(res)}
sc = [r for r in menu if r["scored"]]
hum = [r["human"] for r in sc if r["human"] is not None]
rep["menu"] = {"selected": len(menu), "scored": len(sc), "hard": sum(not r["any"] for r in sc),
               "T_human": len(hum), "human_closed": sum(h["closed"] for h in hum),
               "exclusions": dict(Counter(r["reason"] for r in menu if not r["scored"])),
               "hard_files": len({next(x["file"] for x in reg if x["id"] == r["id"]) for r in sc if not r["any"]})}
rep["per_spec"] = {k: {**charges(v), "fates": fates(v)} for k, v in res.items()}
rep["closure_curves"] = {p: {b: rep["per_spec"].get(tag(p, b), {}).get("solved") for b in BUDGETS}
                         for p in POLICIES}
union = {}
for b in BUDGETS:
    arms = [res[tag(a, b)] for a in ARMS if tag(a, b) in res]
    if len(arms) == 4:
        ids = {r["id"] for r in arms[0]}
        union[b] = sum(any(next(x for x in rows if x["id"] == i)["solved"] for rows in arms) for i in ids)
rep["offline_single_arm_union"] = union
rep["portfolio_vs_best_single"] = {}
for b in BUDGETS:
    singles = {a: res[tag(a, b)] for a in ARMS if tag(a, b) in res}
    if not singles:
        continue
    best = max(singles, key=lambda a: sum(r["solved"] for r in singles[a]))
    rep["portfolio_vs_best_single"][b] = {
        "best_single": best, "best_single_solved": sum(r["solved"] for r in singles[best]),
        **{p: compare(res[tag(p, b)], singles[best]) for p in ("rr", "ucb") if tag(p, b) in res}}
rep["depth_and_width"] = {k: compare(res[k], res[tag("s3", 300)]) for k in
                          (tag("s3", 300, 0, 5), tag("s3", 300, 0, 3, True)) if k in res and tag("s3", 300) in res}
rep["memory_tiers"] = {p: {m: {f: rep["per_spec"].get(tag(p, 300, m), {}).get(f) for f in
                               ("solved", "steps", "premise_rpcs", "premise_hb_raw", "cache_hits",
                                "cache_misses", "cache_evictions", "cache_peak_payload_max", "cache_seconds",
                                "elapsed_seconds")} for m in TIERS} for p in ("rr", "ucb")}
rep["memory_tier_identity"] = {
    p: {m: (sorted((r["id"], r["solved"], r["steps"], tuple(r["path"])) for r in res[tag(p, 300, m)]) ==
            sorted((r["id"], r["solved"], r["steps"], tuple(r["path"])) for r in res[tag(p, 300, 0)]))
        for m in TIERS[1:] if tag(p, 300, m) in res and tag(p, 300, 0) in res} for p in ("rr", "ucb")}
best300 = max((tag(a, 300) for a in ARMS if tag(a, 300) in res),
              key=lambda k: rep["per_spec"][k]["solved"], default=None)
rep["best_300_single"] = best300
if best300:
    rep["paired_vs_best_300_single"] = {k: compare(v, res[best300]) for k, v in res.items()}
rep["replay"] = {"credited": sum(r["solved"] for v in res.values() for r in v),
                 "replayed": sum("replay" in r for v in res.values() for r in v if r["solved"])}
if ref08a:
    ident = {}
    for a in ARMS:
        old = {r["id"]: r for r in stage(ref08a / ("cal_" + a))} if (ref08a / ("cal_" + a)).is_dir() else {}
        new = {r["id"]: r for r in res.get(tag(a, 300), [])}
        common = sorted(set(old) & set(new))
        ident[a] = {"common": len(common),
                    "same_solved": sum(old[k]["solved"] == new[k]["solved"] for k in common),
                    "same_steps_and_path": sum(old[k]["steps"] == new[k]["steps"] and old[k]["path"] == new[k]["path"]
                                               for k in common)}
    rep["identity_with_08a_single_arms"] = ident
for f in ("summary.json", "resource.json"):
    if (run / f).exists():
        rep[f[:-5]] = json.loads((run / f).read_text(encoding="utf-8"))
out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print("ANALYSIS", out, rep["specs_complete"], "of", rep["specs_registered"], "specs")
