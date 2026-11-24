"""Inventory every Lean core source file, declaration and exact goal already spent by
OLR-07A, OLR-07B and the Fable OLR-08 referee pilot, and audit a proposed roster against it.

inventory: authenticate the three archives, read the 07B declared roster from source,
           list the pinned Lean source's src/Init/Data files, write the spent inventory.
audit:     compare a proposed roster ({role: [relpath, ...]}, optional roots with
           file/decl/goal) with the inventory: spent-file overlap, SInt access,
           exact-goal and (file, declaration) collisions, and new-file versus new-family status.
Standard library only.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import olr07_reissue_custody as C  # noqa: E402

REFEREE_SHA = "23225d1ddb28e1f91e013d309a119425649e10ca9a2e7ba0e5521c847eb42e7b"
RESERVED_FAMILY = "SInt"
ROLES = ("train", "calibration", "evaluation")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def family(rel: str) -> str:
    parts = rel.split("/")
    # src/Init/Data/<Family>/<File>.lean or src/Init/Data/<Family>.lean
    return parts[3].removesuffix(".lean") if len(parts) > 3 else rel


def roster_07b(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ROSTER" for t in node.targets):
            roster = ast.literal_eval(node.value)
            return {role: [f"src/Init/Data/{fam}/{f}.lean" for fam, fs in fams.items() for f in fs]
                    for role, fams in roster.items()}
    raise ValueError("ROSTER not found in " + str(path))


def referee(path: Path):
    need = sha(path.read_bytes())
    if need != REFEREE_SHA:
        raise ValueError("referee archive bytes differ from the scope-incident record")
    with zipfile.ZipFile(path) as z:
        roots = json.loads(z.read("olr08_referee/results/registry.json"))
        searched = defaultdict(set)
        for n in z.namelist():
            parts = n.split("/")
            if len(parts) == 4 and parts[1] == "results" and parts[3].endswith(".json") and parts[2] != "native":
                searched[parts[2]].add(parts[3].removesuffix(".json"))
    return roots, {k: sorted(v) for k, v in searched.items()}


def inventory(a):
    old7a, c7a = C.original(a.olr07a, "07a")
    old7b, c7b = C.original(a.olr07b, "07b")
    ref, searched = referee(a.referee)
    declared = roster_07b(a.olr07b_source)
    cohorts = {
        "olr07a_roots": {r: sorted({x["file"] for x in old7a if x["role"] == r}) for r in ROLES},
        "olr07b_declared_roster": declared,
        "olr07b_roots": {r: sorted({x["file"] for x in old7b if x["role"] == r}) for r in ROLES},
        "referee_registry": {r: sorted({x["file"] for x in ref if x["role"] == r}) for r in ROLES},
    }
    files = defaultdict(set)
    for name, by_role in cohorts.items():
        for role, fs in by_role.items():
            for f in fs:
                files[f].add(f"{name}:{role}")
    # Referee per-file logs (clean receipts and search arms) are named <Family>_<File>.
    stems = {f"{family(x['file'])}_{Path(x['file']).stem}": x["file"] for x in ref}
    unmapped = sorted({s for v in searched.values() for s in v} - set(stems))
    for arm, names in searched.items():
        for s in names:
            if s in stems:
                files[stems[s]].add(f"referee_logs:{arm}")
    goals = {sha(x["goal"].encode()) for x in (*old7a, *old7b, *ref)}
    decls = sorted({(x["file"], x["decl"]) for x in (*old7a, *old7b, *ref)})
    rev = subprocess.check_output([a.git, "-C", str(a.source), "rev-parse", "HEAD"], text=True).strip()
    if rev != C.SOURCE:
        raise ValueError("Lean source checkout is not at the pinned commit")
    src = a.source / "src" / "Init" / "Data"
    pinned = sorted(p.relative_to(a.source).as_posix() for p in src.rglob("*.lean"))
    fam_roles = defaultdict(set)
    for f, tags in files.items():
        fam_roles[family(f)].update(t.split(":")[1] for t in tags if not t.startswith("referee_logs:"))
    report = {
        "schema": "olr08.spent_cohorts.v1",
        "inputs": {"olr07a": c7a, "olr07b": c7b, "referee_sha256": REFEREE_SHA,
                   "olr07b_roster_source_sha256": sha(a.olr07b_source.read_bytes()), "lean_source_commit": rev},
        "cohorts": cohorts, "referee_logs": searched, "referee_logs_unmapped": unmapped,
        "spent_files": {f: sorted(t) for f, t in sorted(files.items())},
        "spent_families": {k: sorted(v) for k, v in sorted(fam_roles.items())},
        "spent_goal_sha256": sorted(goals), "spent_decls": [list(d) for d in decls],
        "reserved_family": RESERVED_FAMILY,
        "reserved_files": [f for f in pinned if family(f) == RESERVED_FAMILY],
        "reserved_touched": sorted(f for f in files if family(f) == RESERVED_FAMILY),
        "pinned_init_data_files": len(pinned),
        "unspent_files_by_family": {fam: sorted(f for f in pinned if family(f) == fam and f not in files)
                                    for fam in sorted({family(f) for f in pinned})},
        "counts": {"spent_files": len(files), "spent_goals": len(goals), "spent_decls": len(decls),
                   "roots": {"olr07a": len(old7a), "olr07b": len(old7b), "referee": len(ref)}},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("SPENT", json.dumps(report["counts"]), "reserved_touched", report["reserved_touched"])


def audit(a):
    inv = json.loads(a.inventory.read_text(encoding="utf-8"))
    roster = json.loads(a.roster.read_text(encoding="utf-8"))
    spent = inv["spent_files"]
    fams = inv["spent_families"]
    out = {"schema": "olr08.roster_audit.v1", "inventory_sha256": sha(a.inventory.read_bytes()),
           "roster_sha256": sha(a.roster.read_bytes()), "roles": {}}
    for role, fs in roster.get("files", roster).items():
        if role == "roots" or not isinstance(fs, list) or not all(isinstance(f, str) for f in fs):
            continue
        out["roles"][role] = {
            "files": len(fs),
            "spent_overlap": {f: spent[f] for f in fs if f in spent},
            "reserved_access": [f for f in fs if family(f) == inv["reserved_family"]],
            "new_file_seen_family": sorted(f for f in fs if f not in spent and family(f) in fams),
            "new_family": sorted(f for f in fs if family(f) not in fams),
        }
    roots = roster.get("roots", [])
    if roots:
        goals = set(inv["spent_goal_sha256"])
        decls = {tuple(d) for d in inv["spent_decls"]}
        out["root_collisions"] = {
            "roots": len(roots),
            "exact_goal": sorted(r["id"] for r in roots if sha(r["goal"].encode()) in goals),
            "file_decl": sorted(r["id"] for r in roots if (r["file"], r["decl"]) in decls),
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("ROSTER_AUDIT", json.dumps({r: {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in x.items()}
                                      for r, x in out["roles"].items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inventory")
    i.add_argument("--olr07a", type=Path, required=True, help="original 07A archive (artifact 14890402416)")
    i.add_argument("--olr07b", type=Path, required=True, help="results/olr07b/olr07b-full_AUDIT.zip")
    i.add_argument("--referee", type=Path, required=True, help="olr08_referee.zip")
    i.add_argument("--olr07b-source", type=Path, required=True, help="ppt_gym/olr07b_cpu.py")
    i.add_argument("--source", type=Path, required=True, help="lean4 checkout at the pinned commit")
    i.add_argument("--git", default="git")
    i.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("audit")
    r.add_argument("--inventory", type=Path, required=True)
    r.add_argument("--roster", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    inventory(a) if a.cmd == "inventory" else audit(a)


if __name__ == "__main__":
    main()
