"""Pre-outcome audit of a frozen OLR-08B fresh-calibration registry. Standard library only.

Checks every root against: the 69-file spent inventory (files, exact goals, (file, declaration)
keys, declaration names); the original OLR-08A registry, all roles including evaluation; any
number of other registries (for example a parallel cohort); SInt; the registered roster; source
bytes at the pinned checkout; and internal uniqueness. Writes one JSON report.

Usage: python audit_registry.py <run dir> <spent_cohorts.json> <08A registry.json> <roster.json>
       <lean source dir> <out json> [other registry.json ...]
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

run, inv_p, a08_p, roster_p, source, out = map(Path, sys.argv[1:7])
others = [Path(p) for p in sys.argv[7:]]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def gh(goal):
    return sha(goal.encode())


reg = json.loads((run / "registry.json").read_text(encoding="utf-8"))
seal = (run / "registry.sha256").read_text().strip()
assert sha((run / "registry.json").read_bytes()) == seal, "registry seal"
inv = json.loads(inv_p.read_text(encoding="utf-8"))
a08 = json.loads(a08_p.read_text(encoding="utf-8"))
roster = json.loads(roster_p.read_text(encoding="utf-8"))
spent_files = set(inv["spent_files"]) | set(inv["reserved_files"])
spent_goals = set(inv["spent_goal_sha256"])
spent_fd = {tuple(x) for x in inv["spent_decls"]}
spent_names = {d for _, d in inv["spent_decls"]}
res = {"schema": "olr08b.registry_audit.v1", "registry_sha256": seal, "roots": len(reg),
       "files": len({r["file"] for r in reg}), "roster_sha256": sha(roster_p.read_bytes()),
       "inventory_sha256": sha(inv_p.read_bytes()), "olr08a_registry_sha256": sha(a08_p.read_bytes())}
res["internal"] = {
    "duplicate_ids": len(reg) - len({r["id"] for r in reg}),
    "duplicate_goals": len(reg) - len({gh(r["goal"]) for r in reg}),
    "duplicate_decls": len(reg) - len({r["decl"] for r in reg}),
    "non_calibration": sum(r["role"] != "calibration" for r in reg),
    "outside_roster": sum(r["file"] not in roster for r in reg),
    "sint": sum("SInt" in Path(r["file"]).parts or r["file"].endswith("SInt.lean") for r in reg),
    "source_bytes_mismatch": sum(sha((source / r["file"]).read_bytes()) != r["file_sha256"] for r in reg)}
res["vs_spent_inventory"] = {
    "file": sum(r["file"] in spent_files for r in reg), "goal": sum(gh(r["goal"]) in spent_goals for r in reg),
    "file_decl": sum((r["file"], r["decl"]) in spent_fd for r in reg),
    "decl_name": sum(r["decl"] in spent_names for r in reg)}
res["vs_olr08a_registry_all_roles"] = {
    "roles": dict(Counter(r["role"] for r in a08)),
    "file": sum(r["file"] in {x["file"] for x in a08} for r in reg),
    "goal": sum(gh(r["goal"]) in {gh(x["goal"]) for x in a08} for r in reg),
    "decl_name": sum(r["decl"] in {x["decl"] for x in a08} for r in reg),
    "id": sum(r["id"] in {x["id"] for x in a08} for r in reg)}
res["vs_other_registries"] = []
for p in others:
    o = json.loads(p.read_text(encoding="utf-8"))
    res["vs_other_registries"].append({
        "path": str(p), "sha256": sha(p.read_bytes()), "rows": len(o), "roles": dict(Counter(x["role"] for x in o)),
        "file": sum(r["file"] in {x["file"] for x in o} for r in reg),
        "goal": sum(gh(r["goal"]) in {gh(x["goal"]) for x in o} for r in reg),
        "decl_name": sum(r["decl"] in {x["decl"] for x in o} for r in reg),
        "id": sum(r["id"] in {x["id"] for x in o} for r in reg)})
counts = [*res["internal"].values(), *res["vs_spent_inventory"].values(),
          *(v for k, v in res["vs_olr08a_registry_all_roles"].items() if k != "roles"),
          *(v for o in res["vs_other_registries"] for k, v in o.items() if k in ("file", "goal", "decl_name", "id"))]
res["clean"] = all(c == 0 for c in counts)
res["by_family"] = dict(Counter(r["family"] for r in reg))
res["by_set"] = dict(Counter(r["set"] for r in reg))
out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
print("AUDIT clean=%s" % res["clean"], json.dumps({k: res[k] for k in ("internal", "vs_spent_inventory")}))
