"""Verify a pinned native BPD source tree before entering a child-free worker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


BPD_REVISION = "722613a80d80af0e80bb9f5ac1ce83eef111e4f7"
PACKAGE = "mechinterp/bpd"


def inventory(checkout: Path) -> dict[str, str]:
    root = checkout.resolve()
    package = root / PACKAGE
    if not package.is_dir() or package.resolve() != package:
        raise ValueError("native BPD package missing or redirected")
    result = {}
    for p in sorted(package.rglob("*.py")):
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(package):
            raise ValueError("redirected native BPD source")
        result[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    if not result or f"{PACKAGE}/__init__.py" not in result:
        raise ValueError("incomplete native BPD source tree")
    return result


def preflight(checkout: Path, receipt: Path, revision: str = BPD_REVISION) -> None:
    """Supervisor-only Git verification; worker receives a content manifest."""
    root = checkout.resolve()
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args], timeout=30)
    if git("rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError("native BPD revision mismatch")
    tracked = {p.decode() for p in git("ls-tree", "-r", "--name-only", "-z", "HEAD", "--", PACKAGE).split(b"\0")
               if p.endswith(b".py")}
    files = inventory(root)
    if set(files) != tracked:
        raise ValueError("native BPD Python source roster differs from HEAD")
    # Read the committed blobs, so a dirty tracked source cannot be blessed
    # by merely hashing the current working tree.
    for path, actual in files.items():
        if hashlib.sha256(git("show", f"HEAD:{path}")).hexdigest() != actual:
            raise ValueError(f"native BPD source differs from HEAD: {path}")
    with receipt.open("x", encoding="utf-8") as out:
        json.dump({"schema": "molt-bpd-source-v1", "revision": revision, "files": files}, out, sort_keys=True)


def verify(checkout: Path, receipt: Path, revision: str = BPD_REVISION) -> None:
    """Child-free recheck of the supervisor's trusted source receipt."""
    report = json.loads(receipt.read_text(encoding="utf-8"))
    if (report.get("schema") != "molt-bpd-source-v1" or report.get("revision") != revision
            or report.get("files") != inventory(checkout)):
        raise ValueError("native BPD source receipt mismatch")


def load_verified(checkout: Path, receipt: Path):
    import mechinterp

    verify(checkout, receipt)
    root = checkout.resolve()
    mechinterp.__path__.append(str(root / "mechinterp"))
    import mechinterp.bpd as bpd
    if not Path(bpd.__file__).resolve().is_relative_to(root / PACKAGE):
        raise ValueError("native BPD import outside verified package")
    return bpd
