"""Public synthetic PDI transfer preflight. No held-out or model evidence."""
from __future__ import annotations

import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time

ARMS = ("no_reuse", "verified_reuse", "matched_retrieval", "sham_reuse", "revoked_version")
ROOT_NAME = "pdiRootV1"


def sha(data):
    return sha256(data.encode("utf-8")).hexdigest()


def chain_term(depth, term="h"):
    for i in range(depth):
        term = f"(r{i} {term})"
    return term


def binders(depth):
    props = " ".join(f"P{i}" for i in range(depth + 1))
    rules = " ".join(f"(r{i} : P{i} → P{i+1})" for i in range(depth))
    return props, rules


def root_declaration(depth):
    props, rules = binders(depth)
    return (f"theorem {ROOT_NAME} {{{props} : Prop}} {rules} : P0 → P{depth} := by\n"
            f"  intro h\n  exact {chain_term(depth)}\n")


def route(depth, branch, arm):
    """Breadth-first symbolic search on a declared public graph.

    The supplied root edge is permitted only in verified_reuse. The matched
    retrieval arm has no additional relevant artifact in this empty library.
    """
    if depth < 2 or branch < 0 or arm not in ARMS:
        raise ValueError("invalid task")
    edges = [(f"P{i}", f"P{i+1}", f"r{i}") for i in range(depth)]
    edges += [(f"P{depth}", f"Q{branch}", "b")]
    if arm == "verified_reuse":
        edges += [("P0", f"P{depth}", ROOT_NAME)]
    queue = deque([("P0", "h")])
    seen = {"P0"}
    expanded = 0
    while queue:
        node, term = queue.popleft()
        if node == f"Q{branch}":
            return term, expanded
        expanded += 1
        for a, b, rule in edges:
            if a == node and b not in seen:
                seen.add(b)
                if rule == ROOT_NAME:
                    args = " ".join(f"r{i}" for i in range(depth))
                    new_term = f"({rule} {args} {term})"
                else:
                    new_term = f"({rule} {term})"
                queue.append((b, new_term))
    raise RuntimeError("declared synthetic graph has no route")


def render(depth, branch, arm, *, probe=None):
    props, rules = binders(depth)
    term, expansions = route(depth, branch, arm)
    prefix = "namespace PDIPublicSmoke\n"
    if arm == "verified_reuse":
        prefix += root_declaration(depth)
    elif arm == "sham_reuse":
        prefix += "theorem pdiSham {U V : Prop} (r : U → V) : U → V := r\n"
    name = "pdiTarget"
    if probe == "stale":
        term = f"(b ({ROOT_NAME} " + " ".join(f"r{i}" for i in range(depth)) + " h))"
    elif probe == "self":
        term = name
    elif probe is not None:
        raise ValueError("unknown probe")
    source = (prefix + f"theorem {name} {{{props} Q{branch} : Prop}} "
              f"(h : P0) {rules} (b : P{depth} → Q{branch}) : Q{branch} := by\n"
              f"  exact {term}\n"
              "end PDIPublicSmoke\n"
              "#print axioms PDIPublicSmoke.pdiTarget\n")
    return source, term, expansions


def compile_fresh(lean, out, label, source, timeout):
    path = out / f"{label}.lean"
    log = out / f"{label}.log"
    path.write_text(source, encoding="utf-8")
    start = time.perf_counter()
    try:
        p = subprocess.run([lean, str(path.resolve())], capture_output=True,
                           text=True, encoding="utf-8", timeout=timeout)
        output, code = p.stdout + p.stderr, p.returncode
        status = "pass" if code == 0 and (
            "'PDIPublicSmoke.pdiTarget' does not depend on any axioms" in output
        ) else "rejected"
    except subprocess.TimeoutExpired:
        output, code, status = "compiler timeout", None, "timeout"
    log.write_text(output, encoding="utf-8")
    return {"status": status, "returncode": code, "compiler_ms": round(1000*(time.perf_counter()-start)),
            "source_sha256": sha(source), "log_sha256": sha(output), "source_bytes": len(source.encode("utf-8"))}


def run(lean, out, depths, timeout):
    version = subprocess.check_output([lean, "--version"], text=True, timeout=timeout).strip()
    if "version 4.19.0" not in version:
        raise RuntimeError(f"expected pinned Lean 4.19.0, got {version}")
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for depth in depths:
        for branch in range(4):
            for arm in ARMS:
                source, term, expansions = render(depth, branch, arm)
                result = compile_fresh(lean, out, f"d{depth}_q{branch}_{arm}", source, timeout)
                rows.append({"depth": depth, "target": branch, "arm": arm, "symbolic_expansions": expansions,
                             "proof_chars": len(term), **result})
    sentinels = {}
    for probe in ("stale", "self"):
        source, _, _ = render(depths[0], 0, "revoked_version", probe=probe)
        sentinels[probe] = compile_fresh(lean, out, f"sentinel_{probe}", source, timeout)
    ok = all(r["status"] == "pass" for r in rows) and all(
        r["status"] == "rejected" and r["returncode"] is not None for r in sentinels.values())
    receipt = {"schema": "PDI-public-smoke-v1", "evidence": "public synthetic compiler smoke; no model or held-out data",
               "lean_version": version, "rows": rows, "sentinels": sentinels, "passed": ok,
               "notes": "Fresh compiler process per target. Root elaboration repeated and charged in compiler time; no amortized speed claim."}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", default="lean")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    if len(set(args.depths)) != len(args.depths) or min(args.depths) < 2:
        parser.error("unique depths >= 2 required")
    raise SystemExit(0 if run(args.lean, args.out, args.depths, args.timeout) else 1)
