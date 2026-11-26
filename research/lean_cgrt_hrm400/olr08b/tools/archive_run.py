"""Archive an OLR-08B run directory without rewriting its bytes. Standard library only.

Verifies every stage checkpoint against its .sha256 seal, writes <name>_MANIFEST.json (SHA-256 of
every file in the run directory) next to the archive, and stores the run directory plus the
manifest in <name>_AUDIT.zip. The run directory itself is not modified.

Usage: python archive_run.py <run dir> <dest dir> <name> [extra file ...]
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

run, dest, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
extras = [Path(p) for p in sys.argv[4:]]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


bad = [str(p) for p in run.rglob("*.json")
       if p.with_suffix(".sha256").exists() and p.with_suffix(".sha256").read_text().strip() != sha(p)]
if bad:
    raise SystemExit("seal mismatch: " + ", ".join(bad[:5]))
files = {p.relative_to(run).as_posix(): sha(p) for p in sorted(run.rglob("*")) if p.is_file()}
extra = {"extra/" + p.name: sha(p) for p in extras}
dest.mkdir(parents=True, exist_ok=True)
man = dest / f"{name}_MANIFEST.json"
man.write_text(json.dumps({"run": str(run), "files": files, "extra": extra}, indent=1, sort_keys=True) + "\n",
               encoding="utf-8", newline="\n")
z = dest / f"{name}_AUDIT.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
    for rel in files:
        f.write(run / rel, "run/" + rel)
    for p in extras:
        f.write(p, "extra/" + p.name)
    f.write(man, man.name)
with zipfile.ZipFile(z) as f:
    assert f.testzip() is None
    for rel, h in files.items():
        assert hashlib.sha256(f.read("run/" + rel)).hexdigest() == h
print(z, z.stat().st_size, sha(z), len(files), "files")
