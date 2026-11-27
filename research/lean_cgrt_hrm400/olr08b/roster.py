"""OLR-08B uncontaminated calibration roster builder, exclusion audit, and cohort protection.

Freezes deterministic pre-outcome file selection and exclusion rules BEFORE collecting
any menu, human-replay, or search outcomes.

Hard protection invariants:
  1. Exclude ALL 69 spent files from results/olr08_spent/spent_cohorts.json (07A, 07B, referee).
  2. Exclude ALL 5 calibration files spent in OLR-08A.
  3. Protect ALL 27 declared evaluation files from OLR-08A (original 275-root evaluation
     registry), explicitly flagging smoke-observed root BitVec.shiftLeftZeroExtend
     (id 45ce790b9a0f67f50253ba1b41c778473b4b7631d3a9eef91f6ff0e5642e9a1f) as excluded
     from pristine confirmation (274 pristine evaluation roots).
  4. Keep ALL SInt files strictly sealed (0 SInt files touched).
  5. Keep calibration family-disjoint from TRAIN families (Nat, List, Option, Char, String).
  6. Exclude all spent exact-goal SHA-256 hashes and spent declaration names.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

OLR08_DIR = Path(__file__).resolve().parent.parent / "olr08"
if str(OLR08_DIR) not in sys.path:
    sys.path.insert(0, str(OLR08_DIR))

import probe  # noqa: E402
import run08  # noqa: E402

SEED_08B = 20270211
TRAIN_FAMILIES = frozenset({"Nat", "List", "Option", "Char", "String"})
RESERVED_FAMILY = "SInt"
SMOKE_EVAL_ROOT_ID = "45ce790b9a0f67f50253ba1b41c778473b4b7631d3a9eef91f6ff0e5642e9a1f"
SMOKE_CAL_ROOT_ID = "0c711bff5d411c0d0615cd35d84bf4baabe885a1d0a3234bb9c48de87743d170"

# Frozen pre-outcome calibration file selection rule:
# All unspent, non-SInt, non-OLR08A-cal, non-OLR08A-eval, non-TRAIN-family Lean files
# under src/Init/ (including src/Init/Data/ and src/Init/ core subpackages) that have
# >= 14 syntactic tactic-proof blocks (`:= by` or ` by\n`), ordered lexicographically.
MIN_BY_BLOCKS_RULE = 14
CAP_T_PER_FILE = 12
CAP_R_PER_FILE = 6

# Explicit frozen calibration file list matching the deterministic rule on Lean v4.19.0 (6caaee84):
FROZEN_CAL_FILES_08B: tuple[str, ...] = (
    # src/Init/Data (non-TRAIN families, unspent, non-eval08a, non-SInt)
    "src/Init/Data/AC.lean",
    "src/Init/Data/Array/Bootstrap.lean",
    "src/Init/Data/Array/Lex/Lemmas.lean",
    "src/Init/Data/Bool.lean",
    "src/Init/Data/Int/Bitwise/Lemmas.lean",
    "src/Init/Data/Int/DivMod/Bootstrap.lean",
    "src/Init/Data/Int/DivMod/Lemmas.lean",
    "src/Init/Data/Ord.lean",
    "src/Init/Data/Range/Lemmas.lean",
    "src/Init/Data/Vector/Basic.lean",
    # src/Init core libraries (all brand-new families, unspent, non-SInt)
    "src/Init/Control/Lawful/Basic.lean",
    "src/Init/Control/Lawful/Instances.lean",
    "src/Init/GetElem.lean",
    "src/Init/Grind/Lemmas.lean",
    "src/Init/Grind/Norm.lean",
    "src/Init/Grind/Offset.lean",
    "src/Init/Internal/Order/Basic.lean",
    "src/Init/Internal/Order/Lemmas.lean",
    "src/Init/Omega/Constraint.lean",
    "src/Init/Omega/Int.lean",
    "src/Init/Omega/IntList.lean",
    "src/Init/Omega/LinearCombo.lean",
    "src/Init/PropLemmas.lean",
    "src/Init/SimpLemmas.lean",
    "src/Init/SizeOfLemmas.lean",
    "src/Init/WF.lean",
)


def file_family(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) >= 4 and parts[1] == "Init" and parts[2] == "Data":
        return parts[3].removesuffix(".lean")
    if len(parts) >= 3 and parts[1] == "Init":
        return "Init." + parts[2].removesuffix(".lean")
    if len(parts) >= 3 and parts[1] == "Std":
        return "Std." + parts[2].removesuffix(".lean")
    return parts[-1].removesuffix(".lean")


def load_exclusion_sets(research_dir: Path) -> dict[str, Any]:
    """Build complete contamination-exclusion inventory from all prior OLR artifacts."""
    spent_inv = run08.read(research_dir / "results" / "olr08_spent" / "spent_cohorts.json")
    pkg_spent = run08.read(research_dir / "olr08" / "spent.json")
    reg08a = run08.read(research_dir / "results" / "olr08a" / "preoutcome" / "registry.json")

    spent_files = set(spent_inv["spent_files"]) | set(pkg_spent["files"])
    cal08a_files = {f"src/Init/Data/{f}/{n}.lean" for f, ns in run08.FRESH["calibration"].items() for n in ns}
    eval08a_files = {f"src/Init/Data/{f}/{n}.lean" for f, ns in run08.FRESH["evaluation"].items() for n in ns}
    reserved_sint = set(spent_inv["reserved_files"])

    # Spent goals and declarations include 07A/07B/referee AND OLR-08A spent TRAIN+CAL
    spent_goals = set(spent_inv["spent_goal_sha256"]) | set(pkg_spent["goals_sha256"])
    spent_decls = {d for _, d in spent_inv["spent_decls"]} | set(pkg_spent["decls"])
    for r in reg08a:
        if r["role"] in ("train", "calibration"):
            spent_goals.add(hashlib.sha256(r["goal"].encode("utf-8")).hexdigest())
            spent_decls.add(r["decl"])

    # Protected original 275-root evaluation registry
    eval08a_roots = [r for r in reg08a if r["role"] == "evaluation"]
    pristine_eval_roots = [r for r in eval08a_roots if r["id"] != SMOKE_EVAL_ROOT_ID]

    return {
        "spent_files_07_referee": spent_files,
        "spent_files_08a_cal": cal08a_files,
        "protected_files_08a_eval": eval08a_files,
        "reserved_sint_files": reserved_sint,
        "all_forbidden_cal_files": spent_files | cal08a_files | eval08a_files | reserved_sint,
        "spent_goals_sha256": spent_goals,
        "spent_decls": spent_decls,
        "protected_eval_roots_total": len(eval08a_roots),
        "protected_eval_smoke_excluded_ids": [SMOKE_EVAL_ROOT_ID],
        "protected_eval_pristine_roots": len(pristine_eval_roots),
        "protected_eval_registry_sha256": run08.sha(
            research_dir / "results" / "olr08a" / "preoutcome" / "registry.json"
        ),
    }


def verify_rule_matches_source(source: Path, research_dir: Path, cal_files: tuple[str, ...] = FROZEN_CAL_FILES_08B) -> dict[str, Any]:
    """Audit frozen calibration file list against source checkout and ALL exclusion sets."""
    ex = load_exclusion_sets(research_dir)
    forbidden = ex["all_forbidden_cal_files"]

    # Recompute the static syntactic selection rule directly from source to verify determinism
    recomputed: list[str] = []
    for p in sorted((source / "src" / "Init").rglob("*.lean")):
        rel = p.relative_to(source).as_posix()
        if rel in forbidden:
            continue
        if "/SInt" in rel or rel.endswith("/SInt.lean"):
            continue
        fam = file_family(rel)
        if fam in TRAIN_FAMILIES:
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        nby = txt.count(":= by") + txt.count(" by\n")
        if nby >= MIN_BY_BLOCKS_RULE and rel not in ("src/Init/Core.lean", "src/Init/Tactics.lean"):
            # Core.lean / Tactics.lean define foundational syntax/tactic macros before Init.Data;
            # exclude foundational prelude/tactic-macro files cleanly by explicit static rule.
            recomputed.append(rel)

    cal_set = set(cal_files)
    overlap_spent = sorted(cal_set & ex["spent_files_07_referee"])
    overlap_cal08a = sorted(cal_set & ex["spent_files_08a_cal"])
    overlap_eval08a = sorted(cal_set & ex["protected_files_08a_eval"])
    overlap_sint = sorted(f for f in cal_files if "/SInt" in f or f.endswith("/SInt.lean") or f in ex["reserved_sint_files"])
    overlap_train_fam = sorted(f for f in cal_files if file_family(f) in TRAIN_FAMILIES)
    missing = sorted(f for f in cal_files if not (source / f).is_file())

    passed = (
        not overlap_spent
        and not overlap_cal08a
        and not overlap_eval08a
        and not overlap_sint
        and not overlap_train_fam
        and not missing
        and len(cal_files) == len(cal_set)
    )
    return {
        "schema": "olr08b.roster_audit.v1",
        "passed": passed,
        "cal_files_count": len(cal_files),
        "cal_families": sorted({file_family(f) for f in cal_files}),
        "overlap_spent_07_referee": overlap_spent,
        "overlap_cal_08a": overlap_cal08a,
        "overlap_protected_eval_08a": overlap_eval08a,
        "overlap_sealed_SInt": overlap_sint,
        "overlap_train_families": overlap_train_fam,
        "missing_files": missing,
        "rule_recomputed_count": len(recomputed),
        "rule_exact_match": sorted(cal_files) == sorted(recomputed),
        "rule_diff_extra": sorted(set(cal_files) - set(recomputed)),
        "rule_diff_missing": sorted(set(recomputed) - set(cal_files)),
        "protected_eval_roots_total": ex["protected_eval_roots_total"],
        "protected_eval_pristine_roots": ex["protected_eval_pristine_roots"],
        "protected_eval_smoke_excluded_ids": ex["protected_eval_smoke_excluded_ids"],
        "protected_eval_registry_sha256": ex["protected_eval_registry_sha256"],
        "sealed_SInt_opened": False,
    }


def freeze_olr08b_registry(
    source: Path,
    exe: Path,
    research_dir: Path,
    out: Path,
    *,
    profile: str = "full",
    workers: int = 2,
    cal_files: tuple[str, ...] = FROZEN_CAL_FILES_08B,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically freeze TRAIN (reused referee source) + fresh OLR-08B CAL + protected EVAL manifest.

    Zero menu or search outcomes are consulted during freezing.
    """
    audit = verify_rule_matches_source(source, research_dir, cal_files)
    if not audit["passed"]:
        raise ValueError("OLR-08B roster contamination audit failed: " + json.dumps(audit))

    ex = load_exclusion_sets(research_dir)
    seen_goals = set(ex["spent_goals_sha256"])
    seen_decls = set(ex["spent_decls"])

    # 1. Reused TRAIN cohort (identical deterministic 192 roots as OLR-08A, or 8 for smoke)
    train_all = run08.read(research_dir / "olr08" / "train_registry.json")
    train_rows = sorted(train_all, key=lambda r: run08.digest([run08.SEED, r["id"]]))[: (8 if profile == "smoke" else 192)]

    # 2. Protected original 275-root evaluation registry (preserved verbatim, smoke root flagged!)
    reg08a = run08.read(research_dir / "results" / "olr08a" / "preoutcome" / "registry.json")
    eval_rows = []
    for r in reg08a:
        if r["role"] == "evaluation":
            rec = dict(r)
            rec["pristine_confirmation_eligible"] = r["id"] != SMOKE_EVAL_ROOT_ID
            rec["smoke_observed_in_08a"] = r["id"] == SMOKE_EVAL_ROOT_ID
            eval_rows.append(rec)

    # 3. Fresh uncontaminated OLR-08B calibration files
    target_cal_files = cal_files[:2] if profile == "smoke" else cal_files
    cap_t = 2 if profile == "smoke" else CAP_T_PER_FILE
    cap_r = 1 if profile == "smoke" else CAP_R_PER_FILE

    def trace_one(rel: str) -> tuple[str, Path, bool, str]:
        fam = file_family(rel)
        safe = rel.replace("/", "_").replace(".lean", "")
        d = out / "trace" / safe
        d.mkdir(parents=True, exist_ok=True)
        path = source / rel
        with run08.Lean(exe) as L:
            meta = L.request(op="trace", path=str(path.resolve()), out=str((d / "trace.jsonl").resolve()), cand_every=0)
        if not meta.get("ok") or meta.get("errors"):
            return rel, d, False, str(meta.get("errors") or meta.get("error"))[:240]
        run08.write(d / "report.json", meta)
        return rel, d, True, ""

    cal_rows: list[dict[str, Any]] = []
    trace_census: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rel, d, ok, err in pool.map(trace_one, target_cal_files):
            fam = file_family(rel)
            if not ok:
                trace_census.append({"file": rel, "family": fam, "trace_ok": False, "error": err, "selected": 0, "candidates": 0})
                continue
            rep = run08.read(d / "report.json")
            valid = {
                r["decl"]
                for r in rep["decls"]
                if r["in_env"] and set(r["axioms"]) <= probe.AXIOMS and r["decl"] not in ("", "_anonymous", "[anonymous]")
            }
            allrows = probe.rows_of(probe.read_lines(d / "trace.jsonl"))
            by: dict[str, list[dict]] = defaultdict(list)
            for r in allrows:
                by[r["decl"]].append(r)

            candidates: list[dict[str, Any]] = []
            for decl, rs in by.items():
                if decl not in valid or decl in seen_decls:
                    continue
                first = min(rs, key=lambda r: (r["line"], r["col"]))
                for r in rs:
                    text = r.get("tactic", "")
                    gs = r.get("goals_before", [])
                    if (
                        len(gs) != 1
                        or not text
                        or "…" in text
                        or "…" in gs[0]
                        or probe.FORBIDDEN.search(text)
                        or re.match(r"\s*(decreasing_by|termination_by|where)\b", text)
                    ):
                        continue
                    kind = "T" if r.get("goals_after") == [] else ("R" if r is first else None)
                    if not kind:
                        continue
                    g = gs[0]
                    ident = run08.digest([SEED_08B, rel, decl, r["line"], r["col"], g])
                    candidates.append(
                        {
                            "id": ident,
                            "set": kind,
                            "role": "calibration",
                            "family": fam,
                            "file": rel,
                            "decl": decl,
                            "line": r["line"],
                            "col": r["col"],
                            "goal": g,
                            "human": probe.human_text(r),
                            "decl_rows": len(rs),
                        }
                    )

            selected: list[dict[str, Any]] = []
            for kind, lim in (("T", cap_t), ("R", cap_r)):
                for r in sorted((v for v in candidates if v["set"] == kind), key=lambda x: x["id"]):
                    if sum(v["set"] == kind for v in selected) >= lim:
                        break
                    gh = hashlib.sha256(r["goal"].encode("utf-8")).hexdigest()
                    if gh in seen_goals or r["decl"] in seen_decls:
                        continue
                    seen_goals.add(gh)
                    seen_decls.add(r["decl"])
                    selected.append(r)
            cal_rows.extend(selected)
            trace_census.append(
                {
                    "file": rel,
                    "family": fam,
                    "trace_ok": True,
                    "selected": len(selected),
                    "selected_T": sum(v["set"] == "T" for v in selected),
                    "selected_R": sum(v["set"] == "R" for v in selected),
                    "candidates": len(candidates),
                }
            )

    rows = [*train_rows, *cal_rows, *eval_rows]
    for r in rows:
        r["file_sha256"] = run08.sha(source / r["file"])
    run08.write(out / "registry.json", rows)
    run08.write(out / "registry_census.json", trace_census)
    audit["trace_census_summary"] = {
        "files_attempted": len(target_cal_files),
        "files_traced_ok": sum(1 for c in trace_census if c["trace_ok"]),
        "files_with_roots": sum(1 for c in trace_census if c["selected"] > 0),
        "cal_roots_selected": len(cal_rows),
        "cal_roots_T": sum(v["set"] == "T" for v in cal_rows),
        "cal_roots_R": sum(v["set"] == "R" for v in cal_rows),
        "train_roots_selected": len(train_rows),
        "protected_eval_roots": len(eval_rows),
    }
    run08.write(out / "roster_audit.json", audit)
    return rows, audit
