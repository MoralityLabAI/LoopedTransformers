"""Repository-relocation adapter for the immutable DEUTERIUM-FOLLOWUP-01 source.

This file changes the *provenance check* for an engineering replay after moving
repositories. The scientific steering implementation remains byte-identical to
the pre-scoring registration. Never describe a rerun of the published held-out
seeds as another prospective experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PORT = HERE.parents[1]
REGISTRATION = HERE / "registration.json"
PINNED_UPSTREAM = "2df0092d5be4d8e4748b3c66ef8325dedef17549"
PINNED_REGISTRATION = "f1e3e9ba663ea906a6468d76426bb7381a6e23ad"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git(directory: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def frozen_source_check(commit: str | None, protocol_path: Path) -> dict:
    """Validate immutable bytes and upstream source before opening held-out data."""
    if commit != PINNED_REGISTRATION:
        raise ValueError("Expected the exact frozen --registration-commit")
    registration = json.loads(REGISTRATION.read_text())
    if registration["registration_commit"] != commit:
        raise ValueError("Registration metadata and expected SHA differ")
    if registration["source_repository_base"] != PINNED_UPSTREAM:
        raise ValueError("Registration does not identify the historical source")
    relative_files = {
        "experiments/deuterium/steering.py": HERE / "steering.py",
        "experiments/deuterium/protocol.json": HERE / "protocol.json",
        "experiments/deuterium/requirements-steering.txt": HERE / "requirements-steering.txt",
    }
    for original_path, local_path in relative_files.items():
        if sha256(local_path) != registration["source_sha256"][original_path]:
            raise ValueError(f"Frozen source byte mismatch: {original_path}")
    if protocol_path.resolve() != relative_files["experiments/deuterium/protocol.json"].resolve():
        raise ValueError("Unregistered alternative protocol path")
    upstream_env = os.environ.get("DEUTERIUM_UPSTREAM")
    if not upstream_env:
        raise ValueError("Missing DEUTERIUM_UPSTREAM; original implementation checkout required")
    upstream = Path(upstream_env).resolve()
    if git(upstream, "rev-parse", "HEAD") != PINNED_UPSTREAM:
        raise ValueError("Upstream parameter-decomposition commit mismatch")
    if git(upstream, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Upstream scientific source has tracked modifications")
    module_location = Path(__import__("param_decomp").__path__[0]).resolve()
    if not module_location.is_relative_to(upstream / "param_decomp"):
        raise ValueError("The imported param_decomp is not from the pinned upstream checkout")
    if git(PORT, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Relocated runner requires a clean tracked worktree")
    if git(PORT, "ls-files", "--others", "--exclude-standard", "--", ":(top)mechinterp/deuterium"):
        raise ValueError("Untracked files inside the relocated experiment tree")
    return {
        "registration_commit": commit,
        "protocol_sha256": sha256(protocol_path),
        "source_sha256": {key: sha256(path) for key, path in relative_files.items()},
        "source_commit": git(PORT, "rev-parse", "HEAD"),
        "historical_implementation_commit": PINNED_UPSTREAM,
        "registration_location": "pre-scoring local Git commit in the source repository",
        "mode": "repository-relocation engineering replay; held-out seeds are public",
    }


def main() -> None:
    upstream_env = os.environ.get("DEUTERIUM_UPSTREAM")
    if upstream_env:
        sys.path.insert(0, str(Path(upstream_env).resolve()))
    from experiments.deuterium import steering
    steering.registration_check = frozen_source_check
    steering.main()


if __name__ == "__main__":
    main()
