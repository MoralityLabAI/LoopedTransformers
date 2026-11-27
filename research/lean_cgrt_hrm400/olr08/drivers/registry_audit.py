"""Audit a frozen OLR-08A registry against the full spent-cohort inventory before any
outcome is collected. Fresh (calibration, evaluation) roots must not collide with a spent
file, exact goal, (file, declaration) key or declaration name, and must not touch SInt.
Reused TRAIN roots must come from the package's train_registry.json.

Usage: python registry_audit.py <run dir> <spent_cohorts.json> <audit dir> <package dir> <out json>
"""
import hashlib
import json
import sys
import types
from pathlib import Path

run, inv_path, audit_dir, pkg, out = map(Path, sys.argv[1:6])
sys.path.insert(0, str(audit_dir))
import olr08_spent_cohorts as S  # noqa: E402

reg = json.loads((run / "registry.json").read_text(encoding="utf-8"))
seal = (run / "registry.sha256").read_text().strip()
assert hashlib.sha256((run / "registry.json").read_bytes()).hexdigest() == seal, "registry seal"
fresh = [r for r in reg if r["role"] != "train"]
train = [r for r in reg if r["role"] == "train"]
roster = {"calibration": sorted({r["file"] for r in fresh if r["role"] == "calibration"}),
          "evaluation": sorted({r["file"] for r in fresh if r["role"] == "evaluation"}),
          "roots": fresh}
rp = out.with_name(out.stem + "_roster.json")
rp.write_text(json.dumps(roster, indent=1, sort_keys=True) + "\n", encoding="utf-8")
S.audit(types.SimpleNamespace(inventory=inv_path, roster=rp, out=out))
res = json.loads(out.read_text(encoding="utf-8"))
inv = json.loads(inv_path.read_text(encoding="utf-8"))
names = {d for _, d in inv["spent_decls"]}
pool = {r["id"] for r in json.loads((pkg / "train_registry.json").read_text(encoding="utf-8"))}
res["registry_sha256"] = seal
res["decl_name_collisions"] = sorted(r["id"] for r in fresh if r["decl"] in names)
res["train_roots"] = len(train)
res["train_outside_package_pool"] = sorted(r["id"] for r in train if r["id"] not in pool)
res["fresh_roots_by_role_set"] = {f"{role}:{k}": sum(r["role"] == role and r["set"] == k for r in fresh)
                                  for role in ("calibration", "evaluation") for k in ("T", "R")}
res["fresh_roots_by_family"] = {f: sum(r["family"] == f for r in fresh) for f in sorted({r["family"] for r in fresh})}
res["fresh_files_with_roots"] = len({r["file"] for r in fresh})
bad = (any(x["spent_overlap"] or x["reserved_access"] for x in res["roles"].values())
       or res["root_collisions"]["exact_goal"] or res["root_collisions"]["file_decl"]
       or res["decl_name_collisions"] or res["train_outside_package_pool"])
res["clean"] = not bad
out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print("REGISTRY_AUDIT clean=%s" % res["clean"], json.dumps({k: res[k] for k in (
    "train_roots", "fresh_roots_by_role_set", "fresh_files_with_roots", "decl_name_collisions")}))
