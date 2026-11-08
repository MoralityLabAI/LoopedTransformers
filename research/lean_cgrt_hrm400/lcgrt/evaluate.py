"""Evaluate a checkpoint by Lean-backed proof search and certify every proof it finds.

    python -m lcgrt.evaluate --ckpt D:/lean_cgrt/runs/c1/S_t8_s17 --policies greedy dfs best cgrt cgrt_shuf \
        --splits test_id test_depth test_family test_dep --out results/c1/S_t8_s17.json

Success is counted only when the found tactic script passes `lcgrt.certify` (standalone
Lean file, axiom audit) for the dataset statement. Budget = expansions (Lean calls on a
node). One run at the largest budget (4 x shortest proof + 4) yields success at every
smaller budget; reported: `b1` = shortest + 1, `b2` = 2 x shortest, `b4` = 4 x shortest + 4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from . import config
from . import fragment as F
from .certify import certify, decl_name
from .dataset import iter_records
from .lean_repl import LeanRepl
from .model import ModelConfig, ProofNet
from .search import BRANCH_FEATURES, Scorer, run_searches
from .train import load_verified, sha256

BUDGETS = {"b1": lambda s: s + 1, "b2": lambda s: 2 * s, "b4": lambda s: 4 * s + 4}


def load_model(run_dir: Path, device, allow_incomplete=False):
    if not allow_incomplete and not (Path(run_dir) / "complete.json").exists():
        raise FileNotFoundError(f"{run_dir}: training not complete (no complete.json); refusing to evaluate")
    _, ck = load_verified(Path(run_dir), device)
    if ck is None:
        raise FileNotFoundError(f"no verified checkpoint in {run_dir}")
    cfg = ModelConfig(**ck["config"])
    m = ProofNet(cfg).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck


def eval_items(data, split, limit=None, per_depth=None):
    rows = []
    for r in iter_records(data, split):
        rows.append((r["tid"], r["statement"], {"split": split, "depth": r["depth"], "family": r["family"],
                                                "size": r["proof"]["size"], "seed": r["seed"],
                                                "arity3": r["arity3"], "stmt_sha256": r["stmt_sha256"],
                                                "gen_kwargs": r.get("gen_kwargs")}))
    if per_depth:
        by = {}
        for it in rows:
            by.setdefault(it[2]["depth"], []).append(it)
        rows = [it for d in sorted(by) for it in by[d][:per_depth]]
    return rows[:limit] if limit else rows


def certify_results(results):
    items = []
    for r in results:
        if r["success"]:
            kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (r.get("gen_kwargs") or {}).items()}
            thm = F.generate(r["seed"], r["depth"], r["family"], arity3=r["arity3"], tid=r["tid"], **kw)
            if F.stmt_sha(thm) != r["stmt_sha256"]:
                raise AssertionError(f"{r['tid']}: regenerated statement differs from dataset")
            items.append((r["tid"], thm, r["script"]))
    rec = certify(items) if items else {"status": {}, "source_sha256": None, "lean_version": None, "passed": 0}
    for r in results:
        r["certified"] = bool(r["success"]) and rec["status"].get(decl_name(r["tid"])) == "passed"
    return {k: rec[k] for k in ("source_sha256", "lean_version", "passed")} | {"submitted": len(items)}


def summarize(results):
    out = {}
    for key in ["all"] + sorted({f"depth{r['depth']}" for r in results}):
        rs = results if key == "all" else [r for r in results if f"depth{r['depth']}" == key]
        row = {"n": len(rs)}
        for b, f in BUDGETS.items():
            row[b] = round(sum(r["certified"] and r["expansions"] <= f(r["size"]) for r in rs) / max(1, len(rs)), 4)
        row["mean_expansions_solved"] = round(float(np.mean([r["expansions"] for r in rs if r["certified"]] or [0])), 2)
        out[key] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run directory holding latest.pt/lkg.pt")
    ap.add_argument("--data", default="lcgrt-v1")
    ap.add_argument("--splits", nargs="+", default=["test_id", "test_depth", "test_family", "test_dep"])
    ap.add_argument("--policies", nargs="+", default=["greedy", "dfs", "best"])
    ap.add_argument("--fast-steps", type=int, default=None, help="HRM inference recurrence (default: trained)")
    ap.add_argument("--halt", type=float, default=None, help="adaptive halting threshold")
    ap.add_argument("--branch-weights", default=None, help="JSON from lcgrt.branch for cgrt")
    ap.add_argument("--hide-history", action="store_true", help="policy does not see the dead-goal history")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-depth", type=int, default=None)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cuda-memory-fraction", type=float, default=0.0,
                    help="cap torch's CUDA allocator (Windows spills past VRAM into RAM instead of failing); 0 = off")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.device.startswith("cuda") and a.cuda_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(a.cuda_memory_fraction)
    model = None
    ck_sha = None
    if any(p not in ("random_dfs", "bfs") for p in a.policies):
        model, ck = load_model(Path(a.ckpt), a.device)
        ck_sha = json.loads((Path(a.ckpt) / "latest.json").read_text())["sha256"]
    base_T = model.cfg.fast_steps if model is not None and model.cfg.arch == "hrm" else None
    T = a.fast_steps or base_T
    contexts = [(T, False), (2 * T, False), (T, True)] if base_T else [(None, False), (None, True)]
    bw = json.loads(Path(a.branch_weights).read_text())["weights"] if a.branch_weights else None
    receipt = {"ckpt": str(a.ckpt), "ckpt_sha256": ck_sha, "model": model.cfg.to_json() if model else None,
               "fast_steps": T, "halt": a.halt, "hide_history": a.hide_history, "contexts": contexts, "branch_weights": bw,
               "data_manifest_sha256": sha256(config.DATA_ROOT / a.data / "manifest.json"),
               "budgets": {"b1": "shortest+1", "b2": "2*shortest", "b4": "4*shortest+4"}, "runs": {}}
    out_dir = None
    out = Path(a.out)
    if out.suffix != ".json":            # queue job: a directory with eval.json + complete.json
        out_dir = out; out_dir.mkdir(parents=True, exist_ok=True); out = out_dir / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with LeanRepl() as repl:
        receipt["lean_version"] = repl.version
        for split in a.splits:
            items = eval_items(a.data, split, a.limit, a.per_depth)
            for pol in a.policies:
                t0 = time.time(); c0 = repl.calls
                scorer = Scorer(model, a.device, fast_steps=T, halt_threshold=a.halt) if model else None
                res = run_searches(repl, items, pol, scorer,
                                   budget_fn=lambda m: BUDGETS["b4"](m["size"]),
                                   contexts=contexts, branch_w=bw if pol in ("cgrt", "cgrt_shuf") else None,
                                   width=a.width, seed=zlib.crc32(f"{split}/{pol}".encode()),
                                   hide_history=a.hide_history)
                cert = certify_results(res)
                run = {"split": split, "policy": pol, "summary": summarize(res), "certificate": cert,
                       "seconds": round(time.time() - t0, 1), "lean_calls": repl.calls - c0,
                       "model_forwards": scorer.forwards if scorer else 0,
                       "model_gflops": round(scorer.flops / 1e9, 2) if scorer else 0,
                       "mean_fast_steps": round(float(np.mean(scorer.steps_used)), 2) if scorer and scorer.steps_used else None,
                       "rows": [{k: r[k] for k in ("tid", "depth", "family", "size", "success", "certified",
                                                   "expansions", "lean_attempts")} for r in res]}
                receipt["runs"][f"{split}/{pol}"] = run
                s = run["summary"]["all"]
                print(json.dumps({"split": split, "policy": pol, **s, "seconds": run["seconds"],
                                  "certified": cert["passed"], "submitted": cert["submitted"]}), flush=True)
                out.write_text(json.dumps(receipt, indent=1) + "\n")
    if out_dir is not None:
        (out_dir / "complete.json").write_text(json.dumps({"eval_sha256": sha256(out)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
