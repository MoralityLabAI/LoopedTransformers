"""Audit 03 - Replication from archived checkpoints, per-root budget curves, and the missing
joint (envelope x matrix) factorial intervention.

What it does (calibration cohort only; no training; no sealed data; results/ never written):

  A. Re-runs every registered symbolic control and all 8 learned architectures at B = 128
     from the archived, hash-verified checkpoints and compares (closed, work, path) with
     results/lbf17/episodes.jsonl.gz record by record.
  B. Stores PER-ROOT outcomes at B in {16, 32, 64, 96, 128} for the arms that decide the
     registered recurrence screen (2A) and attribution claim.  The archive only kept
     aggregated budget curves, so AUC contrasts could not be given root-paired intervals.
  C. Adds the unregistered joint intervention `env_mat_zero` (envelope banks AND matrix bank
     zeroed at k >= 2).  The registered module is not edited: a patched copy is generated in
     the scratch directory by three asserted string substitutions, and is checked to be
     output-identical to the registered module on all registered modes before use.
  D. Records, per decision, how many frontier words / pairs were fed to the network and how
     many concrete one-step successors those words have.  Pass 1 of the envelope is exactly
     the abstraction of each frontier word's full successor set, so this is the amount of
     one-hop expansion the controller performs without being charged successor work.

Rows are appended to a JSONL shard; run with --shard i/n for parallel execution, then --merge.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

BUDGETS = (16, 32, 64, 96, 128)
SYMBOLIC = ("bidir", "astar", "geom_mm", "graph_bridge", "exact_lookahead")
BUDGET_ARCHS = ("rel_prov_opened", "param_nonrecurrent", "compute_nonrecurrent",
                "rel_prov_standard", "sym_prov_readout")
SHAMS = ("env_zero", "mat_zero", "env_mat_zero")


def make_patched_module(repo: Path, scratch: Path):
    src_path = common.lbf_dir(repo) / "lbf17_relational_fresco.py"
    src = src_path.read_text(encoding="utf-8")
    subs = [
        ('    "mat_roll",\n)\n', '    "mat_roll",\n    "env_mat_zero",\n)\n'),
        ('                    if mode == "env_zero":\n',
         '                    if mode in ("env_zero", "env_mat_zero"):\n'),
        ('                    elif mode == "mat_zero":\n                        mm_in = torch.zeros_like(mm_in)\n',
         '                    if mode in ("mat_zero", "env_mat_zero"):\n                        mm_in = torch.zeros_like(mm_in)\n'),
    ]
    for old, new in subs:
        if src.count(old) != 1:
            raise ValueError(f"patch anchor not unique: {old!r}")
        src = src.replace(old, new)
    dst = Path(scratch) / "lbf17_audit_patched.py"
    dst.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("lbf17_audit_patched", dst)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, common.sha256_file(src_path)


def select_action(outs, s, m0, m1):
    """Identical to the learned branch of lbf17_relational_fresco.run_lbf17_episode."""
    last = outs[-1]
    psig = last["pot"].sigmoid()
    pm = float(psig.mean())
    s.prev_pot = {(u, v): float(psig[i, j]) - pm
                  for i, u in enumerate(s.fronts[0])
                  for j, v in enumerate(s.fronts[1])}
    s0_l = last["s0"].tolist()
    s1_l = last["s1"].tolist()
    if any(not math.isfinite(x) for x in s0_l + s1_l):
        raise ValueError("nonfinite learned node scores")
    cands = [(s0_l[i], -m0, 0, -i) for i in range(m0)] + [(s1_l[j], -m1, -1, -j) for j in range(m1)]
    best = max(cands)
    return (0 if best[2] == 0 else 1), -best[3]


def clean_episode(rg, m15, c, model, arch: str, mode: str, budget: int, depth: int = 16) -> dict:
    """Uninstrumented episode: no probes, no FLOP counter, no extra forward passes."""
    s = m15.FringeState.start(c)
    snapshots = []
    calls = 0
    t0 = time.perf_counter()
    while s.live(budget):
        m0, m1 = len(s.fronts[0]), len(s.fronts[1])
        snapshots.append((tuple(s.fronts[0]), tuple(s.fronts[1])))
        rem = budget - s.work
        with torch.no_grad():
            if arch == "lbf16_pos1_opened":
                outs, _ = model(c, s, rem, mode="normal", depth_cap=depth)
            else:
                outs, _ = model(c, s, rem, mode=mode, depth_cap=depth)
        calls += 1
        side, idx = select_action(outs, s, m0, m1)
        m15.expand_node(rg, c, s, side, idx, budget, depth)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    res = rg._result(c, "learned", s.path, s.work, s.expanded, calls, budget, depth)

    deg = {}
    def d(w):
        if w not in deg:
            deg[w] = len(rg.neighbors(c, w))
        return deg[w]
    words_seen, pairs_seen = set(), set()
    word_evals = pair_evals = implicit_recomputed = 0
    for fu, fv in snapshots:
        word_evals += len(fu) + len(fv)
        pair_evals += len(fu) * len(fv)
        implicit_recomputed += sum(d(w) for w in fu) + sum(d(w) for w in fv)
        words_seen.update((0, w) for w in fu)
        words_seen.update((1, w) for w in fv)
        pairs_seen.update((u, v) for u in fu for v in fv)
    distinct_words = {w for _, w in words_seen}
    return {
        "closed": res["closed"], "work": res["work"], "expanded": res["expanded"], "path": res["path"],
        "model_calls": calls, "wall_clock_ms_clean": wall_ms,
        "word_evals": word_evals, "distinct_words": len(distinct_words),
        "pair_evals": pair_evals, "distinct_pairs": len(pairs_seen),
        "implicit_onehop_successors_recomputed": implicit_recomputed,
        "implicit_onehop_successors_cached": sum(d(w) for w in distinct_words),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pinned-sim", required=True)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--max-roots", type=int, default=48)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        rows = []
        for f in sorted(out_dir.glob("audit_03_rows.shard*.jsonl")):
            rows.extend(json.loads(x) for x in f.read_text().splitlines() if x)
        rows.sort(key=lambda r: (r["root"], r["condition"], r["seed"] or 0, r["budget"]))
        (out_dir / "audit_03_rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        print("merged", len(rows), "rows")
        return

    si, sn = (int(x) for x in args.shard.split("/"))
    repo = Path(args.repo)
    scratch = common.scratch_dir(args.scratch)
    common.set_single_thread()
    rg, m15, m16, m17 = common.load_modules(repo, Path(args.pinned_sim))
    m17p, src_sha = make_patched_module(repo, scratch)
    cases, _ = common.load_open_split(repo, rg, "calibration")
    cases = cases[:args.max_roots]

    archive = {}
    for r in common.read_jsonl_gz(common.lbf_dir(repo) / "results" / "lbf17" / "episodes.jsonl.gz"):
        archive[(r["root"], r["condition"], r["seed"], r["stream"])] = r

    models, modelsp = {}, {}
    for seed in m17.SEEDS:
        for arch in m17.ALL_LEARNED_ARCHS:
            models[(seed, arch)], _ = common.load_checkpoint(repo, m17, seed, arch, scratch)
        modelsp[seed], _ = common.load_checkpoint(repo, m17p, seed, "rel_prov_opened", scratch)

    # Patched module must be output-identical to the registered module on registered modes.
    patch_checks = 0
    for c in cases[:6]:
        s = m15.FringeState.start(c)
        for _ in range(3):
            if not s.live(128):
                break
            (sd, ix), _, _ = m15.geom_mm_select(rg, c, s)
            m15.expand_node(rg, c, s, sd, ix)
        if not s.live(128):
            continue
        for md in ("normal", "env_zero", "mat_zero", "env_roll", "mat_roll"):
            with torch.no_grad():
                a, _ = models[(107, "rel_prov_opened")](c, s, 128 - s.work, mode=md)
                b, _ = modelsp[107](c, s, 128 - s.work, mode=md)
            for oa, ob in zip(a, b):
                for key in ("pot", "s0", "s1"):
                    if not torch.equal(oa[key], ob[key]):
                        raise AssertionError(f"patched module differs from registered module in mode {md}")
            patch_checks += 1

    shard_path = out_dir / f"audit_03_rows.shard{si}of{sn}.jsonl"
    done = set()
    if shard_path.exists():
        for line in shard_path.read_text().splitlines():
            if line:
                r = json.loads(line)
                done.add((r["root"], r["condition"], r["seed"], r["budget"]))
    fh = open(shard_path, "a", encoding="utf-8")

    def emit(row):
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()

    for ci, c in enumerate(cases):
        if ci % sn != si:
            continue
        # A. symbolic controls, registered runner, B = 128
        for cname in SYMBOLIC:
            if (c.root, cname, None, 128) in done:
                continue
            r = m17.run_lbf17_episode(rg, c, cname)
            a = archive[(c.root, cname, None, None)]
            emit({"root": c.root, "condition": cname, "seed": None, "budget": 128,
                  "closed": r["closed"], "work": r["work"], "path_len": len(r["path"]),
                  "archive_closed": a["closed"], "archive_work": a["work"],
                  "matches_archive": bool(r["closed"] == a["closed"] and r["work"] == a["work"] and r["path"] == a["path"])})
        for seed in m17.SEEDS:
            jobs = [(arch, "normal", 128, models[(seed, arch)], arch) for arch in m17.ALL_LEARNED_ARCHS]
            for arch in BUDGET_ARCHS:
                jobs += [(arch, "normal", b, models[(seed, arch)], arch) for b in BUDGETS[:-1]]
            for md in SHAMS:
                jobs += [(f"rel_prov_{md}", md, b, modelsp[seed], "rel_prov_opened") for b in BUDGETS]
            for cond, mode, budget, model, arch in jobs:
                if (c.root, cond, seed, budget) in done:
                    continue
                r = clean_episode(rg, m15, c, model, arch, mode, budget)
                row = {"root": c.root, "condition": cond, "seed": seed, "budget": budget,
                       "closed": r["closed"], "work": r["work"], "path_len": len(r["path"]),
                       **{k: r[k] for k in ("model_calls", "wall_clock_ms_clean", "word_evals", "distinct_words",
                                            "pair_evals", "distinct_pairs",
                                            "implicit_onehop_successors_recomputed",
                                            "implicit_onehop_successors_cached")}}
                a = archive.get((c.root, cond, seed, None)) if budget == 128 else None
                if a is not None:
                    row.update({"archive_closed": a["closed"], "archive_work": a["work"],
                                "archive_model_calls": a["model_calls"],
                                "archive_wall_clock_ms": a["wall_clock_ms"],
                                "matches_archive": bool(r["closed"] == a["closed"] and r["work"] == a["work"]
                                                        and r["path"] == a["path"])})
                emit(row)
        print(f"[audit03 shard {si}/{sn}] root {ci + 1}/{len(cases)} {c.root} done", flush=True)
    fh.close()
    common.write_json(out_dir / f"audit_03_meta.shard{si}of{sn}.json", {
        "registered_module_sha256": src_sha, "patched_vs_registered_identity_checks": patch_checks,
        "torch_version": torch.__version__, "threads": torch.get_num_threads(),
    })


if __name__ == "__main__":
    main()
