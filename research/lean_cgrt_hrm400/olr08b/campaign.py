"""OLR-08B native, development-only control and memory sweep.

Run from a checkout of the stacked branch; the 08A Lean binary is reused.
No evaluation code path or reader fitting in this first rung.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from portfolio import ARMS, BUDGETS, TIERS, ArmReceipt, ExactMenuCache, caps, choose, headroom

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT / "olr08"))
import run08 as old
from search08 import actions, reprice, FREE
from launch import limit_memory

RESULTS = PARENT / "results"
REGISTRY08A = RESULTS / "olr08a/preoutcome/registry.json"
REGISTRY08A_HASH = "d61bf9797c50823cd591472d879085a954b68974334e97f46f964b9d5b38905e"
PRIOR08A = RESULTS / "olr08a/full/prior.json"
INVENTORY = RESULTS / "olr08_spent/spent_cohorts.json"


def authenticated_inputs():
    if old.sha(REGISTRY08A) != REGISTRY08A_HASH:
        raise ValueError("08A original registry hash mismatch")
    inv = old.read(INVENTORY)
    if inv["schema"] != "olr08.spent_cohorts.v1" or len(inv["spent_files"]) != 69:
        raise ValueError("unexpected spent inventory")
    return old.read(REGISTRY08A), inv


def source_check(source):
    def git(*a):
        return subprocess.check_output(["git", "-C", str(source), *a], text=True).strip()
    if git("rev-parse", "HEAD") != old.COMMIT or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("pinned clean Lean source checkout required")


def reserved(path):
    return any(x == "SInt" or x == "SInt.lean" for x in Path(path).parts)


def deny_files(original, inv):
    files = set(inv["spent_files"]) | set(inv["reserved_files"])
    files |= {r["file"] for r in original}
    files |= {f"src/Init/Data/{f}/{n}.lean"
              for fams in old.FRESH.values() for f, names in fams.items() for n in names}
    return files


def validate_roster(files, original, inv, source):
    if not isinstance(files, list) or len(files) < 12 or len(set(files)) != len(files):
        raise ValueError("need >=12 unique prospective calibration files")
    denied = deny_files(original, inv)
    fams = defaultdict(list)
    for f in files:
        if not isinstance(f, str) or "\\" in f:
            raise ValueError("noncanonical roster path")
        p = Path(f)
        if (p.is_absolute() or ".." in p.parts or p.as_posix() != f
                or p.parts[:3] != ("src", "Init", "Data") or len(p.parts) != 5
                or p.suffix != ".lean" or f in denied or reserved(f)
                or not (source / p).is_file()):
            raise ValueError("unsafe, unavailable or spent file: " + f)
        fams[p.parts[3]].append(p.stem)
    return {k: sorted(v) for k, v in sorted(fams.items())}


def seal_registry(out, rows, provenance):
    p = out / "registry.json"
    if p.exists() or (out / "registry.sha256").exists():
        raise ValueError("registry already exists; use a new freeze directory")
    old.write(p, rows)
    (out / "registry.sha256").write_text(old.sha(p) + "\n", encoding="utf-8")
    old.write(out / "registry_provenance.json", provenance)


def freeze_new(args, original, inv):
    if args.roster is None:
        raise ValueError("freeze-new requires an explicit registered file list")
    roster = old.read(args.roster)
    fams = validate_roster(roster, original, inv, args.source)
    scratch = args.out / "trace_only"
    if scratch.exists():
        raise ValueError("freeze requires a fresh trace directory")
    # Reuse the exact 08A pre-command trace selector. It never runs menu outcomes.
    prior_fresh = old.FRESH
    try:
        old.FRESH = {"calibration": fams}
        selected = old.freeze(args.source, args.exe, scratch, "full", args.workers)
    finally:
        old.FRESH = prior_fresh
    seen_goals = set(inv["spent_goal_sha256"])
    seen_goals |= {hashlib.sha256(r["goal"].encode()).hexdigest() for r in original}
    seen_decls = {r["decl"] for r in original} | {d for _, d in inv["spent_decls"]}
    rejected, rows = [], []
    for r in selected:
        if r["role"] != "calibration":
            continue
        h = hashlib.sha256(r["goal"].encode()).hexdigest()
        if h in seen_goals or r["decl"] in seen_decls:
            rejected.append({"id": r["id"], "reason": "prior goal or declaration overlap"})
            continue
        seen_goals.add(h)
        seen_decls.add(r["decl"])
        rows.append(r)
    seal_registry(args.out, rows, {"mode": "fresh-calibration", "roster": roster,
                  "roster_sha256": old.sha(args.roster), "original": REGISTRY08A_HASH,
                  "inventory_sha256": old.sha(INVENTORY), "rejected": rejected})
    old.write(args.out / "freeze_summary.json", {"roots": len(rows),
        "files": len({r["file"] for r in rows}), "menu_outcomes_collected": 0,
        "possible_feasibility": len(rows) >= 120 and len({r["file"] for r in rows}) >= 12,
        "registry_sha256": old.sha(args.out / "registry.json")})


def read_registry(args, original, inv):
    out = args.out
    if args.mode == "spent" and not (out / "registry.json").exists():
        seal_registry(out, [r for r in original if r["role"] == "calibration"],
                      {"mode": "spent", "original": REGISTRY08A_HASH})
    p = out / "registry.json"
    h = old.sha(p)
    if (out / "registry.sha256").read_text().strip() != h:
        raise ValueError("registry seal mismatch")
    if args.mode == "fresh-calibration" and args.expected_registry_sha256 != h:
        raise ValueError("commit and supply frozen registry digest before native outcomes")
    provenance = old.read(out / "registry_provenance.json")
    if provenance["mode"] != args.mode or provenance["original"] != REGISTRY08A_HASH:
        raise ValueError("wrong registry provenance")
    rows = old.read(p)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate root")
    old_cal = {r["id"]: r for r in original if r["role"] == "calibration"}
    denied = deny_files(original, inv)
    old_goals = set(inv["spent_goal_sha256"]) | {
        hashlib.sha256(r["goal"].encode()).hexdigest() for r in original}
    old_decls = {r["decl"] for r in original} | {d for _, d in inv["spent_decls"]}
    new_goals, new_decls = set(), set()
    if args.mode == "fresh-calibration":
        validate_roster(provenance["roster"], original, inv, args.source)
        if provenance["inventory_sha256"] != old.sha(INVENTORY):
            raise ValueError("spent inventory changed")
    for r in rows:
        if r["role"] != "calibration" or reserved(r["file"]):
            raise ValueError("evaluation or reserved root refused")
        if args.mode == "spent":
            if old_cal.get(r["id"]) != r:
                raise ValueError("spent mode requires original calibration rows")
        else:
            hgoal = hashlib.sha256(r["goal"].encode()).hexdigest()
            if (r["file"] in denied or r["file"] not in provenance["roster"]
                    or hgoal in old_goals or hgoal in new_goals
                    or r["decl"] in old_decls or r["decl"] in new_decls):
                raise ValueError("fresh calibration overlap")
            new_goals.add(hgoal)
            new_decls.add(r["decl"])
        if old.sha(args.source / r["file"]) != r["file_sha256"]:
            raise ValueError("source file changed")
    return rows


def menu_for(m, state, cache, wider=False):
    sid = state["sid"]
    found = cache.get(sid)
    if found is not None:
        return found
    # Same behavior as the 08A repair: charge native timeout, keep fixed actions.
    p = m.call("premises", sid=sid, max_rw=64 if wider else 40,
               max_apply=16 if wider else 10)
    timeout = (p.get("ok") is False and "maximum number of heartbeats" in str(p.get("error", ""))
               and "rw" not in p and "apply" not in p)
    if timeout:
        p = {"rw": [], "apply": []}
    if p.get("error") or not all(k in p for k in ("rw", "apply")):
        raise ValueError("native premise failure: " + str(p)[:300])
    seen, out = set(), []
    for a in actions(state["info"], p, k_rw=40 if wider else 24,
                     k_apply=10 if wider else 6):
        if a[-1] not in seen:
            out.append(a)
            seen.add(a[-1])
    if not timeout:
        cache.put(sid, out)
    return out


class Queue:
    def __init__(self, meter, root, arm, prior, cache, depth, wider):
        self.m, self.arm, self.prior, self.cache = meter, arm, prior, cache
        self.depth, self.wider = depth, wider
        self.heap, self.events, self.serial = [], [], 0
        self.path, self.on_path = None, []
        self.expand(root, 0, 0., (), ())

    def expand(self, state, depth, g, path, ids):
        acts = menu_for(self.m, state, self.cache, self.wider)
        costs = reprice(acts, self.prior, depth) if self.arm in ("s3", "s3c") else acts
        if self.arm == "s1":
            costs = [(1., *a[1:]) for a in acts]
        ranks = Counter()
        for c, kind, arg, tactic in costs:
            rank = ranks[kind]
            ranks[kind] += 1
            extra = 0.
            if self.arm == "s3c":
                extra = math.log2(max(1., self.prior["hb"].get(
                    tactic if kind in FREE else kind, self.prior["hb_mean"])))
            self.serial += 1
            heapq.heappush(self.heap, (g+c+extra, self.serial, state["sid"],
                                      depth, g+c, path, ids, kind, arg, tactic, rank))

    def step(self):
        _, _, sid, d, g, path, ids, kind, arg, tac, rank = heapq.heappop(self.heap)
        a = self.m.call("step", sid=sid, tactic=tac, hb=500 if tac == "exact?" else 200)
        e = {"kind": kind, "argument": arg, "tactic": tac, "rank": rank, "depth": d,
             "ok": a.get("ok") is True, "ngoals": a.get("ngoals"),
             "hb_raw": a["request_hb_raw"], "parent_events": list(ids)}
        i = len(self.events)
        self.events.append(e)
        if not e["ok"]:
            return e
        path, ids = (*path, tac), (*ids, i)
        if a["ngoals"] == 0:
            if old.close_cert(self.m.call("cert", sid=a["sid"]).get("cert", {})):
                self.path, self.on_path = list(path), list(ids)
        elif d + 1 < self.depth:
            self.expand(a, d+1, g, path, ids)
        return e


def replay(exe, source, row, path):
    # Extended depth support, otherwise same repaired 08A replay semantics.
    with old.Lean(exe) as L:
        s, why, opening = old.open_root(L, source, row)
        if s is None:
            raise ValueError("replay root unavailable: " + str(why))
        m = old.Meter(L, steps=len(path)+1, rpcs=100, hb=old.HB_CAP)
        warm = []
        for d, tac in enumerate(path):
            if tac == "exact?":
                t = time.perf_counter()
                w = L.request(op="step", sid=s["sid"], tactic=tac, hb=old.REPLAY_WARM_HB)
                warm.append({"depth": d, "hb_raw": w.get("request_hb_raw"),
                             "server_ms": w.get("request_ms"), "ok": w.get("ok"),
                             "wall_seconds": time.perf_counter()-t})
            s = m.call("step", sid=s["sid"], tactic=tac, hb=500 if tac == "exact?" else 200)
            if s.get("ok") is not True:
                raise ValueError("fresh path replay failed")
        if s.get("ngoals") != 0 or not old.close_cert(m.call("cert", sid=s["sid"]).get("cert", {})):
            raise ValueError("fresh certificate failed")
        return {"open": opening, "meter": m.report(), "library_search_warmup": warm}


def search(exe, source, row, prior, policy, budget, mib, depth=3, wider=False, workers=1):
    if row["role"] != "calibration" or reserved(row["file"]):
        raise ValueError("non-calibration root refused")
    if depth not in (3, 5) or (wider and depth != 3):
        raise ValueError("unregistered diagnostic")
    t = time.perf_counter()
    # Aggregate cache payload bound across concurrent workers; separate hard process-tree cap.
    cache = ExactMenuCache(mib)
    cache.cap //= workers
    arms = (policy,) if policy in ARMS else ARMS
    if policy not in (*ARMS, "rr", "ucb"):
        raise ValueError("unregistered policy")
    queues, winner, schedule = {}, None, []
    receipts = {a: ArmReceipt() for a in arms}
    with old.Lean(exe) as L:
        s, why, opening = old.open_root(L, source, row)
        if s is None:
            raise ValueError("scored root unavailable: " + str(why))
        m = old.Meter(L, *caps(budget))
        try:
            for a in arms:
                queues[a] = Queue(m, s, a, prior, cache, depth, wider)
            turn = 0
            while not m.exhausted():
                live = [a for a, q in queues.items() if q.heap]
                if not live:
                    break
                a = arms[0] if len(arms) == 1 else choose(policy, arms, live, receipts, turn)
                if policy == "rr":
                    turn = (arms.index(a)+1) % len(arms)
                q, before = queues[a], m.hb
                event = q.step()
                receipts[a].observe(event, m.hb-before, depth)
                schedule.append(a)
                if q.path is not None:
                    winner = q
                    break
        except old.Exhausted:
            pass
        meter = m.report()
    result = {"id": row["id"], "file": row["file"], "family": row["family"],
              "role": row["role"], "policy": policy, "budget": budget, "memory_mib": mib,
              "depth": depth, "wider": wider, "solved": winner is not None,
              "path": winner.path if winner else [], "on_path": winner.on_path if winner else [],
              "winning_arm": winner.arm if winner else None, "open": opening,
              "meter": meter, "steps": meter["steps"], "cache": cache.report(),
              "schedule": schedule, "receipts": {a: vars(r) for a, r in receipts.items()},
              "queues": {a: q.events for a, q in queues.items()}}
    if winner:
        result["replay"] = replay(exe, source, row, winner.path)
    result["elapsed"] = time.perf_counter()-t
    return result


def validity(rows):
    human = [r["human"] for r in rows if r["scored"] and r["human"] is not None]
    return {"human_n": len(human), "human_closed": sum(h["closed"] for h in human),
            "rate": sum(h["closed"] for h in human)/max(1,len(human))}


def specs(mode):
    # Fresh calibration: core 300-attempt controls, memory sweep and long reference.
    # Spent exploratory: the full budget grid plus depth/action coverage diagnostics.
    budgets = BUDGETS if mode == "spent" else (300,)
    runs = [(p,b,0,3,False) for b in budgets for p in (*ARMS,"rr","ucb")]
    if ("s3",1200,0,3,False) not in runs:
        runs.append(("s3",1200,0,3,False))
    runs += [(p,300,m,3,False) for m in TIERS[1:] for p in ("rr","ucb")]
    if mode == "spent":
        runs += [("s3",300,0,5,False),("s3",300,0,3,True)]
    return runs


def tag(spec):
    p,b,m,d,w = spec
    return f"{p}_b{b}_m{m}_d{d}_w{int(w)}"


def campaign(args, original, inv):
    rows = read_registry(args, original, inv)
    prior = old.read(PRIOR08A)
    runspecs = specs(args.mode)
    contract = {"schema": "olr08b.control.v1", "mode": args.mode,
                "source": old.COMMIT, "binary": old.sha(args.exe),
                "registry": old.sha(args.out/"registry.json"),
                "prior": old.sha(PRIOR08A), "inventory": old.sha(INVENTORY),
                "threads": args.threads, "workers": args.workers, "ram_gib": args.ram_gib,
                "python": sys.version, "specs": [list(s) for s in runspecs],
                "code": {str(p.relative_to(PARENT)): old.sha(p)
                         for folder in (HERE,PARENT/"olr08") for p in folder.glob("*.py")}}
    contract_path = args.out/"contract.json"
    if contract_path.exists() and old.read(contract_path) != contract:
        raise ValueError("resume contract differs")
    old.write(contract_path, contract)
    if args.mode == "fresh-calibration" and (
            len(rows) < 120 or len({r["file"] for r in rows}) < 12):
        old.write(args.out/"summary.json", {"status":"registry_feasibility_failed",
                  "roots":len(rows), "files":len({r["file"] for r in rows}),
                  "evaluation_collected":0, "training_updates":0})
        return
    rec = old.stage(args.out,"menu_cal",rows,lambda r: old.menu(args.exe,args.source,r),args.workers)
    val = validity(rec)
    ids = {r["id"] for r in rec if r["scored"] and not r["any"]}
    hard = [r for r in rows if r["id"] in ids]
    if args.mode == "fresh-calibration" and (val["human_n"] < 10 or val["rate"] < .95
           or len(hard)<120 or len({r["file"] for r in hard})<12):
        old.write(args.out/"summary.json", {"status":"menu_feasibility_failed",
                  "validity":val, "hard_roots":len(hard),
                  "files":len({r["file"] for r in hard}),
                  "evaluation_collected":0,"training_updates":0})
        return
    results = {}
    for s in runspecs:
        name = tag(s)
        results[name] = old.stage(args.out,name,hard,
            lambda r,s=s: search(args.exe,args.source,r,prior,*s,workers=args.workers),args.workers)
    reference = tag(("s3",1200,0,3,False))
    controls = [tag(s) for s in runspecs if s[1]==300 and s[3]==3 and not s[4]]
    gate = headroom(results,reference,controls)
    best = max(controls,key=lambda k:gate["counts"][k])
    summary = {"status":"controls_complete","mode":args.mode,"validity":val,"gate":gate,
               "gate_eligible":args.mode=="fresh-calibration" and gate["passes"],
               "best_control":best,"summary":{k:old.rate(v) for k,v in results.items()},
               "paired_vs_best":{k:old.compare(v,results[best]) for k,v in results.items()},
               "cache":{k:{field:sum(r["cache"][field] for r in v)
                           for field in ("hits","misses","evictions","seconds")} |
                          {"max_root_peak_payload":max((r["cache"]["peak"] for r in v),default=0)}
                        for k,v in results.items()},
               "evaluation_collected":0,"training_updates":0,"SInt_opened":False,
               "architecture_promotion_eligible":False}
    old.write(args.out/"summary.json",summary)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("spent","freeze-new","fresh-calibration"),required=True)
    for a in ("source","exe","out"):
        p.add_argument("--"+a,type=Path,required=True)
    p.add_argument("--roster",type=Path)
    p.add_argument("--expected-registry-sha256")
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--ram-gib",type=float,default=12)
    return p.parse_args()


def main():
    args=cli()
    if not 1 <= args.workers <= args.threads <= 64 or not 0 < args.ram_gib <= 12:
        raise ValueError("invalid CPU/RAM profile")
    os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS=str(args.threads),
                      MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),
                      LEAN_NUM_THREADS=str(max(1,args.threads//args.workers)))
    peak,kind=limit_memory(args.ram_gib)
    import psutil
    proc=psutil.Process()
    if hasattr(proc,"cpu_affinity"):
        proc.cpu_affinity(proc.cpu_affinity()[:args.threads])
    args.source,args.exe,args.out=(p.resolve() for p in (args.source,args.exe,args.out))
    args.out.mkdir(parents=True,exist_ok=True)
    source_check(args.source)
    original,inv=authenticated_inputs()
    # 08A already verified the exact reissue manifests; this campaign binds
    # to its registered original registry and unchanged repaired native code.
    old.sentinel(args.exe,args.out/"sentinel")
    stats={"limit":kind,"ram_gib":args.ram_gib,"threads":args.threads,
           "workers":args.workers,"peak_tree_rss":0,"complete":False}
    done=threading.Event()
    def sample():
        while not done.is_set():
            try:
                stats["peak_tree_rss"]=max(stats["peak_tree_rss"],sum(
                    p.memory_info().rss for p in [proc,*proc.children(recursive=True)]))
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                pass
            done.wait(.5)
    monitor=threading.Thread(target=sample,daemon=True)
    monitor.start()
    t=time.perf_counter()
    try:
        if args.mode=="freeze-new":
            freeze_new(args,original,inv)
        else:
            campaign(args,original,inv)
        stats["complete"]=True
    except BaseException as e:
        stats["error"]=repr(e)
        raise
    finally:
        done.set()
        monitor.join()
        stats["seconds"]=time.perf_counter()-t
        if peak:
            stats["peak_job_commit"]=peak()
        old.write(args.out/"resource.json",stats)


if __name__=="__main__":
    main()
