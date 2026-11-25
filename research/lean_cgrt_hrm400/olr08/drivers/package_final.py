"""Build olr08-full_AUDIT.zip and olr08-full_FABLE.zip from the reported run directory.

Same file selection as launch.py (code/ from the package HOME is already in the run
directory), plus driver/ (the scripts that ran the campaign), analysis/ (report, registry
audit, determinism comparison, stopped-run records, smoke records, results document and
repairs). audit_manifest.json lists the SHA-256 of every other file.

Usage: python package_final.py <run dir> <scripts dir> <repo research dir> <dest dir>
"""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

run, scripts, research, dest = map(Path, sys.argv[1:5])


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


drv = run / "driver"
if drv.exists():
    shutil.rmtree(drv)
drv.mkdir()
for p in sorted([*scripts.glob("*.py"), *scripts.glob("*.sh")]):
    shutil.copy2(p, drv / p.name)
an = run / "analysis"
(an / "repairs").mkdir(parents=True, exist_ok=True)
shutil.copy2(research / "OLR_08A_RESULTS.md", an / "OLR_08A_RESULTS.md")
shutil.copy2(research / "OLR_08A_PREOUTCOME.md", an / "OLR_08A_PREOUTCOME.md")
shutil.copy2(research / "olr08" / "REPAIRS.md", an / "repairs" / "REPAIRS.md")
for p in sorted((research / "olr08" / "repairs").glob("*.patch")):
    shutil.copy2(p, an / "repairs" / p.name)

files = {p.relative_to(run).as_posix(): sha(p) for p in sorted(run.rglob("*"))
         if p.is_file() and p.name != "audit_manifest.json"}
(run / "audit_manifest.json").write_text(json.dumps({"files": files}, ensure_ascii=False, sort_keys=True) + "\n",
                                         encoding="utf-8")
keep = {"summary.json", "contract.json", "registry_census.json", "resource.json", "frozen_policy.json", "prior.json",
        "registry.json", "analysis/report.json", "analysis/determinism.json", "analysis/registry_audit.json",
        "analysis/OLR_08A_RESULTS.md", "analysis/OLR_08A_PREOUTCOME.md", "analysis/attempts.tsv"}
fable = [n for n in files if n.startswith(("code/", "driver/", "reissue_", "analysis/repairs/", "analysis/stopped_runs/"))
         or n in keep]
dest.mkdir(parents=True, exist_ok=True)
for name, names in (("olr08-full_AUDIT.zip", [*files, "audit_manifest.json"]),
                    ("olr08-full_FABLE.zip", [*fable, "audit_manifest.json"])):
    z = dest / name
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for n in names:
            f.write(run / n, n)
    print(z, z.stat().st_size, sha(z), flush=True)
