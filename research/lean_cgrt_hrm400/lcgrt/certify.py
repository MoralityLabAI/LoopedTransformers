"""Certificate of record: compile a standalone Lean file and audit axioms.

The certifier writes the file itself from structured data:
  * the statement comes from the dataset's theorem fields (binding is by construction,
    a proof cannot be attached to a weaker statement);
  * every tactic line must match TACTIC_RE (`apply <hyp>` or `constructor`), so `sorry`,
    `admit`, `native_decide`, `set_option`, `#eval`, macros and axioms cannot appear;
  * a theorem passes only if Lean exits 0, reports no error inside its line range, and
    `#print axioms` for that exact declaration says it depends on no axioms.
Incomplete replay (unsolved goals) is a Lean error, hence a rejection.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

from . import config
from . import fragment as F

TACTIC_RE = re.compile(r"^(apply [hr][0-9]{1,4}|constructor)$")
NAME_RE = re.compile(r"[^A-Za-z0-9_]")


def decl_name(tid: str) -> str:
    return "t_" + NAME_RE.sub("_", tid)


def render(items):
    """items: [(tid, Theorem, script)] -> (source, {name: (first_line, last_line)}, rejected)"""
    lines = ["namespace LcgrtCert", ""]
    spans, rejected = {}, {}
    for tid, thm, script in items:
        bad = [t for t in script if not TACTIC_RE.match(t)]
        if bad or not script:
            rejected[decl_name(tid)] = f"tactic outside whitelist: {bad[:3]}" if bad else "empty script"
            continue
        name = decl_name(tid)
        start = len(lines) + 1
        lines.append(f"theorem {name} {F.binders(thm)} : {F.target(thm)} := by")
        lines += [f"  {t}" for t in script]
        spans[name] = (start, len(lines))
        lines.append("")
    lines.append("end LcgrtCert")
    lines += [f"#print axioms LcgrtCert.{n}" for n in spans]
    return "\n".join(lines) + "\n", spans, rejected


def certify(items, timeout: int = 900, keep: Path | None = None) -> dict:
    source, spans, rejected = render(items)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "Cert.lean"
        path.write_text(source, encoding="utf-8")
        if keep is not None:
            keep.parent.mkdir(parents=True, exist_ok=True)
            keep.write_text(source, encoding="utf-8")
        try:
            p = subprocess.run([str(config.LEAN_EXE), str(path)], capture_output=True, text=True,
                               encoding="utf-8", timeout=timeout, env=config.lean_env())
            out, rc, timed_out = p.stdout + p.stderr, p.returncode, False
        except subprocess.TimeoutExpired:
            out, rc, timed_out = "", None, True
        version = subprocess.run([str(config.LEAN_EXE), "--version"], capture_output=True, text=True,
                                 env=config.lean_env()).stdout.strip()
    clean, error_lines = set(), []
    for line in out.splitlines():
        m = re.match(r"^'LcgrtCert\.([A-Za-z0-9_]+)' does not depend on any axioms$", line.strip())
        if m:
            clean.add(m.group(1))
        m = re.match(r"^.*Cert\.lean:(\d+):\d+: error", line)
        if m:
            error_lines.append(int(m.group(1)))
    status = {}
    for name, (a, b) in spans.items():
        bad = timed_out or any(a <= ln <= b for ln in error_lines)
        status[name] = "passed" if (name in clean and not bad) else "rejected"
    for name, why in rejected.items():
        status[name] = "rejected"
    return {
        "lean_version": version, "lean_exe": str(config.LEAN_EXE), "returncode": rc, "timed_out": timed_out,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "declarations": len(spans), "passed": sum(v == "passed" for v in status.values()),
        "rejected": {k: v for k, v in status.items() if v != "passed"},
        "whitelist_rejections": rejected, "status": status,
        "error_lines": error_lines[:50], "output_tail": out[-3000:],
    }
