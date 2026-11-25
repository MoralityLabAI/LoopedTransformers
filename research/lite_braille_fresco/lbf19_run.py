"""LBF-19 command-line runner: cohort generation, scoring (count / lean modes) and archiving.

Usage (python from the psurg venv):
  lbf19_run.py gen    --out DIR --seed-hex HEX --per-l 32 --label confirm
  lbf19_run.py score  --cohort DIR --mode count|lean --budget B --arms a,b,c --workers W --out DIR [--lean-cap N]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import os
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lbf19_scaled_domain as sd  # noqa: E402

STEPS_RANGE = {16: (70, 110), 20: (45, 70), 24: (28, 50)}


def sha256_file(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------- gen
def _gen_one_length(args):
    seed, L, n, label, excl = args
    cs, st = sd.generate_scaled(seed, L, n, label, STEPS_RANGE[L], exclude_lineages=set(excl), max_attempts=5000)
    return L, [(c.public(), info) for c, info in cs], st


def historical_lineages() -> set[str]:
    out = set()
    for d in (HERE / "results" / "lbf17" / "suite",):
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        for sp in ("train", "calibration", "ordinary", "ood"):
            gz = d / f"{sp}.jsonl.gz"
            raw = d / f"{sp}.jsonl"
            if gz.exists():
                data = gzip.decompress(gz.read_bytes()).decode()
            elif raw.exists():
                data = raw.read_text(encoding="utf-8")
            else:
                continue
            for line in data.splitlines():
                if line.strip():
                    out.add(json.loads(line)["lineage"])
    return out


def cmd_gen(a):
    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit("fresh cohort directory required")
    out.mkdir(parents=True)
    excl = sorted(historical_lineages() | set(a.exclude_lineages_from and _load_lineages(a.exclude_lineages_from) or []))
    seed_int = int(a.seed_hex[:16], 16) if a.seed_hex else a.seed
    jobs = []
    for L in (16, 20, 24):
        sub = int(hashlib.sha256(f"{seed_int}:{L}:{a.label}".encode()).hexdigest()[:12], 16)
        jobs.append((sub, L, a.per_l, a.label, excl))
    rows, stats = [], {}
    with cf.ProcessPoolExecutor(max_workers=3) as ex:
        for L, items, st in ex.map(_gen_one_length, jobs):
            stats[L] = st
            for i, (pub, info) in enumerate(items):
                pub = dict(pub)
                pub["root"] = f"{a.label}_L{L}_{i:03d}_{pub['lineage'][:12]}"
                rows.append({**pub, "classical": info})
    lin = [r["lineage"] for r in rows]
    assert len(set(lin)) == len(lin)
    (out / "cases.jsonl").write_text("".join(sd.canonical(r) + "\n" for r in rows), encoding="utf-8")
    man = {"schema": "lbf19.scaled_cohort.v1", "label": a.label, "seed_int": seed_int, "seed_hex": a.seed_hex,
           "per_l": a.per_l, "lengths": [16, 20, 24], "steps_range": {str(k): v for k, v in STEPS_RANGE.items()},
           "hardness_window_expansions": [sd.MIN_EXP, sd.MAX_EXP], "n_roots": len(rows),
           "generator_stats": {str(k): v for k, v in stats.items()},
           "cases_sha256": sha256_file(out / "cases.jsonl"),
           "excluded_historical_lineages": len(excl),
           "selection_uses_policy_outcomes": "classical uncapped bidir expansions only"}
    (out / "manifest.json").write_text(sd.canonical(man) + "\n", encoding="utf-8")
    print(json.dumps(man, indent=1))


def _load_lineages(path):
    rows = [json.loads(x) for x in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r["lineage"] for r in rows]


def load_cohort(d) -> list[tuple[sd.ScaledCase, dict]]:
    d = pathlib.Path(d)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(d / "cases.jsonl") != man["cases_sha256"]:
        raise SystemExit("cohort hash mismatch")
    out = []
    for x in (d / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            o = json.loads(x)
            out.append((sd.ScaledCase.parse(o), o["classical"]))
    return out


# -------------------------------------------------------------------------------------------- score
def _score_root(task):
    (cdict, classical, arms, mode, budget, lean_cap, out_dir, certify, threads) = task
    import torch
    torch.set_num_threads(threads)
    import lbf19_harness as h
    import lbf19_native_verifier as nv
    case = sd.ScaledCase.parse(cdict)
    rg = h.load_rg()
    bank = h.ModelBank()
    repl = nv.start_repl() if mode == "lean" else None
    recs, tapes = [], {}
    try:
        for arm in arms:
            seeds = h.SEEDS if arm in h.LEARNED_ARMS else (0,)
            for seed in seeds:
                cap = None
                if arm in ("bidir_uncapped", "exact_mm") and mode == "lean":
                    cap = lean_cap
                r = h.run_arm(rg, case, arm, seed=seed, budget=budget, bank=bank, repl=repl, mode=mode,
                              call_cap=cap, certify=(certify and mode == "lean"))
                if mode == "lean" and "cert" in r:
                    tape = r["cert"].pop("tape")
                    name = f"c_{arm}_{seed}_B{budget}"
                    tapes[name] = tape
                    r["cert"]["tape_name"] = name
                recs.append(r)
                if mode == "lean":
                    repl.reset()
        batch = None
        if tapes:
            batch = nv.compile_batch(case, tapes, pathlib.Path(out_dir) / "certs", case.root)
        return {"root": case.root, "records": recs, "cert_batch": batch, "error": None}
    except Exception as e:  # keep partial results, report loudly
        return {"root": case.root, "records": recs, "cert_batch": None,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"}
    finally:
        if repl is not None:
            repl.close()


def cmd_score(a):
    cohort = load_cohort(a.cohort)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = a.arms.split(",")
    tasks = [(c.public(), info, arms, a.mode, a.budget, a.lean_cap, str(out), a.certify, a.threads) for c, info in cohort]
    if a.limit:
        tasks = tasks[:a.limit]
    t0 = time.perf_counter()
    rows, certs, errors = [], [], []
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_score_root, t) for t in tasks]
        for i, f in enumerate(cf.as_completed(futs)):
            r = f.result()
            rows.extend(r["records"])
            if r["cert_batch"]:
                certs.append({"root": r["root"], **r["cert_batch"]})
            if r["error"]:
                errors.append({"root": r["root"], "error": r["error"]})
            print(f"[{i + 1}/{len(tasks)}] {r['root']} {len(r['records'])} recs "
                  f"{'ERR' if r['error'] else 'ok'} {time.perf_counter() - t0:.0f}s", flush=True)
    stem = f"{a.mode}_B{a.budget}"
    with gzip.open(out / f"episodes_{stem}.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(sd.canonical(r) + "\n")
    (out / f"certs_{stem}.json").write_text(json.dumps(certs, indent=1), encoding="utf-8")
    (out / f"errors_{stem}.json").write_text(json.dumps(errors, indent=1), encoding="utf-8")
    print(f"done {len(rows)} episodes, {len(errors)} errors, {time.perf_counter() - t0:.0f}s")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--seed-hex", default="")
    g.add_argument("--per-l", type=int, default=32)
    g.add_argument("--label", default="dev")
    g.add_argument("--exclude-lineages-from", default="")
    s = sub.add_parser("score")
    s.add_argument("--cohort", required=True)
    s.add_argument("--mode", choices=("count", "lean"), default="count")
    s.add_argument("--budget", type=int, default=512)
    s.add_argument("--arms", default="bidir_uncapped,sym_prov_readout")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--out", required=True)
    s.add_argument("--lean-cap", type=int, default=30000)
    s.add_argument("--certify", action="store_true")
    s.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    {"gen": cmd_gen, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
