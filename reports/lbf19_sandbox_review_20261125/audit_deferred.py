"""Post-hoc audit of LBF-19's public archive; does not train or alter historical outcomes."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LBF = HERE.parents[1] / "research" / "lite_braille_fresco"
sys.path.insert(0, str(LBF))
import lbf19_scaled_domain as sd

BASE = LBF / "results" / "lbf19"


def apply(case, word, step):
    i, d, p = step
    a, b = case.rules[i]
    l, r = (a, b) if d == 0 else (b, a)
    if word[p:p + len(l)] != l:
        raise AssertionError("rewrite does not match")
    return word[:p] + r + word[p + len(l):]


def replay(case, steps):
    word = case.source
    for step in steps:
        word = apply(case, word, step)
    assert word == case.target


def term(word, leaf="x"):
    for c in reversed(word):
        leaf = f"f_{c} ({leaf})"
    return leaf


def deferred_proof(case, path):
    """Construct an explicit congruence proof. No search-time native verifier needed."""
    name = "deferred_" + case.root
    binders = ["(A : Type)", "(f_a f_b f_c f_d f_e f_f : A → A)", "(x : A)"]
    binders += [
        f"(h{i} : ∀ z : A, {term(a, 'z')} = {term(b, 'z')})"
        for i, (a, b) in enumerate(case.rules)
    ]
    lines = [
        f"theorem {name} {' '.join(binders)} : {term(case.source)} = {term(case.target)} := by",
        "  calc",
    ]
    word = case.source
    for j, m in enumerate(path):
        i, d, p = m
        a, b = case.rules[i]
        left = a if d == 0 else b
        h = f"(h{i} ({term(word[p + len(left):])}))"
        if d:
            h = f"{h}.symm"
        proof = f"congrArg (fun z : A => {term(word[:p], 'z')}) {h}"
        nxt = apply(case, word, m)
        lines.append(f"    {term(word) if j == 0 else '_'} = {term(nxt)} := {proof}")
        word = nxt
    assert word == case.target
    lines.append(f"#print axioms {name}")
    return "\n".join(lines) + "\n"


def verify_archive(cases, out):
    path = BASE / "confirm_out" / "episodes_lean_B512.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    lookup = {c.root: c for c in cases}
    assert len(rows) == 144
    assert len({(r["root"], r["arm"], r["seed"], r["budget"]) for r in rows}) == len(rows)
    assert {r["root"] for r in rows} == set(lookup)
    assert sum(r["v_verifier_calls"] for r in rows) == 127959
    assert sum(r["v_mismatches"] for r in rows) == 0
    passed = [r for r in rows if r["closed"]]
    for r in passed:
        replay(lookup[r["root"]], r["path"])
    assert len(passed) == 28
    assert all(r["arm"] == "sym_prov_readout" for r in passed)
    certs = json.loads((BASE / "confirm_out" / "certs_lean_B512.json").read_text(encoding="utf-8-sig"))
    assert sum(r["n_theorems"] for r in certs) == len(passed)
    hash_matches = {}
    for r in certs:
        raw = (BASE / "confirm_out" / "certs" / r["file"]).read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        options = {"raw": raw, "LF": lf, "CRLF": lf.replace(b"\n", b"\r\n")}
        hash_matches[r["file"]] = [k for k, v in options.items() if hashlib.sha256(v).hexdigest() == r["sha256"]]
    arms = collections.defaultdict(list)
    for r in rows:
        arms[r["arm"]].append(r)
    # Observed closures by a common call/time budget; right-censored runs are not assumed solved.
    endpoints = {}
    for arm, a in arms.items():
        endpoints[arm] = {
            "episodes": len(a), "roots": len({r["root"] for r in a}),
            "closures": sum(bool(r["closed"]) for r in a),
            "mean_calls": statistics.mean(r["v_verifier_calls"] for r in a),
            "mean_recorded_search_seconds": statistics.mean(r["wall_seconds"] for r in a),
            "mean_init_seconds": statistics.mean(r["v_init_seconds"] for r in a),
            "closure_by_call_cap": {
                str(n): sum(r["closed"] and r["v_verifier_calls"] <= n for r in a) / len(a)
                for n in (128, 256, 512, 1024, 2000)
            },
            "closure_by_search_plus_init_seconds": {
                str(t): sum(r["closed"] and r["wall_seconds"] + r["v_init_seconds"] <= t for r in a) / len(a)
                for t in (5, 10, 20, 30, 50)
            },
        }
    # H1 restricted contrast: exact one-sided sign test for nonnegative nonzero root differences.
    primary = arms["sym_prov_readout"]
    by_root = collections.defaultdict(list)
    for r in primary:
        by_root[r["root"]].append(bool(r["closed"]))
    assert all(len(v) == 3 for v in by_root.values())
    assert not any(r["closed"] for r in arms["geom_mm"])
    nonzero = sum(any(v) for v in by_root.values())
    summary = {
        "archive_episodes": len(rows), "archived_closed_paths_replayed": len(passed),
        "archive_tactic_calls": sum(r["v_verifier_calls"] for r in rows),
        "cert_file_hash_matches": hash_matches, "certificate_files": len(certs),
        "certificate_theorems": sum(r["n_theorems"] for r in certs),
        "arms": endpoints, "H1_vs_geom_only_exact_signflip": 2 ** (-nonzero),
        "registered_graph_bridge_missing": "graph_bridge" not in arms,
        "registered_live_recurrent_missing": "rel_prov_guarded" not in arms,
        "recorded_wall_excludes_init_and_final_certification": True,
    }
    (out / "archive_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--lean", type=pathlib.Path, help="Explicit Lean 4.19.0 executable")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    case_rows = [
        json.loads(l) for l in (BASE / "confirm_cohort" / "cases.jsonl").read_text(encoding="utf-8-sig").splitlines()
        if l.strip()
    ]
    cases = [sd.ScaledCase.parse(r) for r in case_rows]
    assert len(cases) == 24
    archive = verify_archive(cases, out)
    results, sources = [], []
    for c, archived in zip(cases, case_rows):
        t0 = time.perf_counter()
        result = sd.classical_bidir(c, depth=24)
        elapsed = time.perf_counter() - t0
        assert result["closed"]
        expected = archived["classical"]
        counters = {k: result[k] for k in ("expanded", "visited", "work", "path_len")}
        assert all(v == expected["classical_" + k] for k, v in counters.items())
        path = [(m["rule"], m["direction"], m["position"]) for m in result["path"]]
        replay(c, path)
        sources.append(deferred_proof(c, path))
        results.append({"root": c.root, "L": len(c.source), **counters,
                        "search_seconds": elapsed, "proof_steps": path})
        print(c.root, counters, round(elapsed, 3), flush=True)
    source = out / "DeferredProofs.lean"
    source.write_text("\n".join(sources), encoding="utf-8")
    summary = {
        "roots": len(results), "exact_count_matches": len(results), "python_replay_passes": len(results),
        "mean_search_seconds": statistics.mean(r["search_seconds"] for r in results),
        "median_search_seconds": statistics.median(r["search_seconds"] for r in results),
        "max_search_seconds": max(r["search_seconds"] for r in results),
        "mean_path_len": statistics.mean(r["path_len"] for r in results),
        "native_compiled": False, "rows": results,
    }
    if args.lean:
        version = subprocess.check_output([str(args.lean), "--version"], text=True)
        assert "4.19.0" in version, version
        start = time.perf_counter()
        proc = subprocess.run([str(args.lean), "--threads=1", str(source)],
                              capture_output=True, text=True, timeout=600)
        summary["lean_compile_seconds"] = time.perf_counter() - start
        log = proc.stdout + proc.stderr
        (out / "DeferredProofs.log").write_text(log, encoding="utf-8")
        names = re.findall(r"'(deferred_[A-Za-z0-9_]+)' does not depend on any axioms", log)
        assert proc.returncode == 0 and len(set(names)) == 24, log
        summary["native_compiled"] = True
        summary["native_axiom_free_declarations"] = len(set(names))
    (out / "deferred_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print("Archive receipts:", archive["archive_episodes"], archive["archived_closed_paths_replayed"])


if __name__ == "__main__":
    main()
