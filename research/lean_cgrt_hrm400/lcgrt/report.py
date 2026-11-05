"""Collect campaign receipts into the repository and build result tables.

    python -m lcgrt.report --campaign c1

Writes results/<campaign>/: train_receipts.json, evals/<job>.json.gz (full evaluation
receipts with per-theorem rows), branch/<job>.json, summary.json and tables.md. Paired
comparisons use a theorem-level bootstrap over the same held-out theorems (10,000
resamples); seed spread is reported separately where three seeds exist.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from . import config

SPLITS = ("test_id", "test_depth", "test_family", "test_dep", "test_reason", "test_mirror",
          "test_mirror_id", "test_mirror_deep")
BUDGET_KEYS = ("b1", "b2", "b4")
BUDGETS = {"b1": lambda s: s + 1, "b2": lambda s: 2 * s, "b4": lambda s: 4 * s + 4}


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def train_receipt(d: Path):
    cfg = json.loads((d / "config.json").read_text())
    done = json.loads((d / "complete.json").read_text())
    rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    vals = [r for r in rows if "val" in r]
    log = (d / "train.log").read_text(encoding="utf-8", errors="replace") if (d / "train.log").exists() else ""
    pauses = [json.loads(l)["reason"] for l in log.splitlines() if l.startswith('{"event": "pause"')]
    return {"run": d.name, "params": cfg["parameters"]["total"], "model": cfg["model"],
            "identity_hash": cfg["identity_hash"], "identity": cfg["identity"],
            "checkpoint_sha256": done["checkpoint_sha256"], "final_val": vals[-1]["val"] if vals else None,
            "final_loss": rows[-1]["loss"] if rows else None,
            "steps_per_s_median": float(np.median([r["steps_per_s"] for r in rows])) if rows else None,
            "peak_vram_mib": max((r.get("peak_vram_mib", 0) for r in rows), default=None),
            "pauses": len(pauses), "pause_reasons": sorted(set(p.split()[0] for p in pauses)),
            "train_theorems": cfg["train_theorems"], "train_states": cfg["train_states"], "torch": cfg["torch"]}


def solved(rows, b):
    return {r["tid"]: bool(r["certified"] and r["expansions"] <= BUDGETS[b](r["size"])) for r in rows}


def paired_bootstrap(a: dict, b: dict, n=10_000, seed=0):
    keys = sorted(set(a) & set(b))
    x = np.array([a[k] for k in keys], float) - np.array([b[k] for k in keys], float)
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return {"diff": round(float(x.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                                         round(float(np.percentile(boots, 97.5)), 4)], "n": len(keys)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="c1")
    a = ap.parse_args(argv)
    root = config.RUN_ROOT / a.campaign
    out = config.RESULTS / a.campaign
    (out / "evals").mkdir(parents=True, exist_ok=True); (out / "branch").mkdir(parents=True, exist_ok=True)
    trains, evals, branches = {}, {}, {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "complete.json").exists():
            continue
        if (d / "config.json").exists():
            trains[d.name] = train_receipt(d)
        elif (d / "eval.json").exists():
            e = json.loads((d / "eval.json").read_text())
            evals[d.name] = e
            with gzip.open(out / "evals" / f"{d.name}.json.gz", "wt", encoding="utf-8") as f:
                json.dump(e, f)
        elif (d / "branch.json").exists():
            branches[d.name] = json.loads((d / "branch.json").read_text())
            (out / "branch" / f"{d.name}.json").write_text((d / "branch.json").read_text())
    (out / "train_receipts.json").write_text(json.dumps(trains, indent=1) + "\n")

    table = []      # one row per (eval job, split, policy)
    for job, e in evals.items():
        for key, run in e["runs"].items():
            split, pol = key.split("/")
            s = run["summary"]["all"]
            table.append({"job": job, "split": split, "policy": pol, **{b: s[b] for b in BUDGET_KEYS},
                          "n": s["n"], "mean_expansions_solved": s["mean_expansions_solved"],
                          "model_forwards": run["model_forwards"], "model_gflops": run["model_gflops"],
                          "mean_fast_steps": run["mean_fast_steps"], "certified": run["certificate"]["passed"],
                          "lean_calls": run["lean_calls"], "seconds": run["seconds"],
                          "eval_sha256": hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest()[:16],
                          "ckpt_sha256": e.get("ckpt_sha256"), "fast_steps": e.get("fast_steps"), "halt": e.get("halt"),
                          "by_depth": {k: v for k, v in run["summary"].items() if k != "all"}})

    def rows_of(job, split, pol):
        """Rows for (split, policy) from `job`, else from its test_reason follow-up evalR_*."""
        for j in (job, job.replace("eval_", "evalR_", 1), job.replace("eval_", "evalM_", 1),
                  job.replace("evalT16_", "evalMT16_", 1), job.replace("evalT32_", "evalMT32_", 1)):
            e = evals.get(j)
            rows = e["runs"].get(f"{split}/{pol}", {}).get("rows") if e else None
            if rows:
                return rows
        return None

    comparisons = {}
    pairs = [("HRM-S vs flat-S", "eval_S_t8_s17b", "eval_flatS_s17"),
             ("HRM-M vs flat-M", "eval_M_t8_s17", "eval_flatM_s17"),
             ("HRM-L vs flat-L", "eval_L_t8_s17", "eval_flatL_s17"),
             ("HRM-S vs single-clock loop", "eval_S_t8_s17b", "eval_Sloop_s17b"),
             ("HRM-S vs no-graph", "eval_S_t8_s17b", "eval_S_nograph_s17"),
             ("HRM-S T8 vs T1", "eval_S_t8_s17b", "eval_S_t1_s17"),
             ("HRM-S T8 vs T2", "eval_S_t8_s17b", "eval_S_t2_s17"),
             ("HRM-S T8 vs T4", "eval_S_t8_s17b", "eval_S_t4_s17"),
             ("HRM-S vs flat-M (FLOP-matched)", "eval_S_t8_s17b", "eval_flatM_s17"),
             ("c2 HRM-S randdepth vs flat-S (s17)", "eval_c2_Srd_s17", "eval_c2_flatS_s17"),
             ("c2 HRM-S randdepth vs flat-S (s29)", "eval_c2_Srd_s29", "eval_c2_flatS_s29"),
             ("c2 HRM-S randdepth vs HRM-S T8", "eval_c2_Srd_s17", "eval_c2_S_t8_s17")]
    for name, x, y in pairs:
        for split in SPLITS:
            for pol in ("dfs", "best", "greedy"):
                ra, rb = rows_of(x, split, pol), rows_of(y, split, pol)
                if ra and rb:
                    for b in ("b1", "b2"):
                        comparisons[f"{name} | {split} | {pol} | {b}"] = paired_bootstrap(solved(ra, b), solved(rb, b))
    for run in ("S_t8_s17b", "M_t8_s17", "L_t8_s17", "S_t8_s29", "S_t8_s43"):
        for split in SPLITS:
            ra, rb = rows_of(f"evalcgrt_{run}", split, "cgrt"), rows_of(f"evalcgrt_{run}", split, "cgrt_shuf")
            rc = rows_of(f"evalcgrt_{run}", split, "best")
            for b in ("b1", "b2"):
                if ra and rb:
                    comparisons[f"CGRT vs shuffled ({run}) | {split} | {b}"] = paired_bootstrap(solved(ra, b), solved(rb, b))
                if ra and rc:
                    comparisons[f"CGRT vs best-first 1-context ({run}) | {split} | {b}"] = paired_bootstrap(solved(ra, b), solved(rc, b))
    for run in ("S_t8_s17b", "M_t8_s17", "L_t8_s17", "Sloop_s17b", "c2_S_t8_s17", "c2_Srd_s17", "c2_Srd_s29"):
        for T in ("T16", "T32"):
            for split in ("test_depth", "test_reason", "test_mirror", "test_mirror_id", "test_mirror_deep"):
                for pol in ("best", "greedy"):
                    rt, r8 = rows_of(f"eval{T}_{run}", split, pol), rows_of(f"eval_{run}", split, pol)
                    if rt and r8:
                        for b in ("b1", "b2"):
                            comparisons[f"infer {T} vs T8 ({run}) | {split} | {pol} | {b}"] = \
                                paired_bootstrap(solved(rt, b), solved(r8, b))
        for split in SPLITS:
            rh, r0 = rows_of(f"evalnohist_{run}", split, "best"), rows_of(f"eval_{run}", split, "best")
            if rh and r0:
                for b in ("b1", "b2"):
                    comparisons[f"history visible vs hidden ({run}) | {split} | best | {b}"] = \
                        paired_bootstrap(solved(r0, b), solved(rh, b))

    seeds = {}
    for arch, runs in (("HRM-S", ["eval_S_t8_s17b", "eval_S_t8_s29", "eval_S_t8_s43"]),
                       ("flat-S", ["eval_flatS_s17", "eval_flatS_s29", "eval_flatS_s43"])):
        for split in SPLITS:
            for pol in ("dfs", "best", "greedy"):
                for b in ("b1", "b2"):
                    v = [np.mean(list(solved(rows_of(r, split, pol), b).values())) for r in runs if rows_of(r, split, pol)]
                    if len(v) == 3:
                        seeds[f"{arch} | {split} | {pol} | {b}"] = {"mean": round(float(np.mean(v)), 4),
                                                                    "sd": round(float(np.std(v, ddof=1)), 4),
                                                                    "per_seed": [round(float(x), 4) for x in v]}
    summary = {"campaign": a.campaign, "train": trains, "eval_table": table, "comparisons": comparisons,
               "seed_spread": seeds, "branch": branches}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "tables.md").write_text(render(summary), encoding="utf-8")
    print(json.dumps({"train_runs": len(trains), "eval_jobs": len(evals), "branch_jobs": len(branches),
                      "comparisons": len(comparisons)}))


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def render(s):
    """Markdown tables. Success = Lean-certified proof within the budget (b1/b2/b4)."""
    L = []
    L.append("### Training receipts\n")
    L.append("| run | params | val top-1 on choice states (last tick) | steps/s (median) | peak VRAM MiB | thermal/RAM pauses | checkpoint sha256 |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for name, t in sorted(s["train"].items()):
        v = t["final_val"]["choice_top1_by_tick"][-1] if t["final_val"] else None
        L.append(f"| {name} | {t['params']:,} | {pct(v)} | {t['steps_per_s_median']:.2f} | {t['peak_vram_mib']} | "
                 f"{t['pauses']} ({', '.join(t['pause_reasons'])}) | `{t['checkpoint_sha256'][:12]}` |")
    by = {}
    for r in s["eval_table"]:
        by[(r["job"], r["split"], r["policy"])] = r
    jobs = sorted({r["job"] for r in s["eval_table"]})
    for pol in ("best", "dfs", "greedy", "cgrt", "cgrt_shuf", "random_dfs", "bfs"):
        rows = [j for j in jobs if any((j, sp, pol) in by for sp in SPLITS)]
        if not rows:
            continue
        L.append(f"\n### Policy `{pol}`: certified success % at b1 / b2 (n per split in brackets)\n")
        L.append("| eval job | " + " | ".join(SPLITS) + " | GFLOP/theorem (test_depth) | mean fast steps |")
        L.append("|---|" + "---|" * len(SPLITS) + "---:|---:|")
        for j in rows:
            cells = []
            for sp in SPLITS:
                r = by.get((j, sp, pol))
                cells.append(f"{pct(r['b1'])} / {pct(r['b2'])} [{r['n']}]" if r else "–")
            rd = by.get((j, "test_depth", pol))
            gf = f"{rd['model_gflops'] / max(1, rd['n']):.1f}" if rd and rd["model_gflops"] else "–"
            fs = rd["mean_fast_steps"] if rd and rd["mean_fast_steps"] else "–"
            L.append(f"| {j} | " + " | ".join(cells) + f" | {gf} | {fs} |")
    if s["seed_spread"]:
        L.append("\n### Seed spread (3 seeds each)\n")
        L.append("| model / split / policy / budget | mean % | sd | per seed |")
        L.append("|---|---:|---:|---|")
        for k, v in s["seed_spread"].items():
            L.append(f"| {k} | {pct(v['mean'])} | {100 * v['sd']:.1f} | {', '.join(pct(x) for x in v['per_seed'])} |")
    if s["comparisons"]:
        L.append("\n### Paired comparisons (same theorems; theorem-level bootstrap, 10,000 resamples)\n")
        L.append("| comparison | difference (points) | 95% interval | n |")
        L.append("|---|---:|---|---:|")
        for k, v in s["comparisons"].items():
            L.append(f"| {k} | {100 * v['diff']:+.1f} | [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}] | {v['n']} |")
    if s["branch"]:
        L.append("\n### CGRT branch scorer (fit on training theorems, checked on held-aside training theorems)\n")
        L.append("| job | rows | check AUC learned | check AUC prior | check AUC mean log p |")
        L.append("|---|---:|---:|---:|---:|")
        for k, v in sorted(s["branch"].items()):
            L.append(f"| {k} | {v['rows']} | {v['check_auc_learned']} | {v['check_auc_prior_score']} | {v['check_auc_mean_logp_only']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
