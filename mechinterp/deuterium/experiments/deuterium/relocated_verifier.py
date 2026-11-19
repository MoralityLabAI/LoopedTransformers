"""Read-only source provenance adapter for the original independent verifier.

The original verify_followup.py code remains byte-identical. This wrapper
services only its two historical `git show` lookups using the byte-verified
frozen source copies in the relocated repository, and rejects all other git
requests. The evidence verifier still recomputes all held-out arithmetic.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

from experiments.deuterium import relocated_runner, verify_followup


ALLOWED = {
    "experiments/deuterium/protocol.json": relocated_runner.HERE / "protocol.json",
    "experiments/deuterium/steering.py": relocated_runner.HERE / "steering.py",
}


def historical_git_show(argv: list[str], **kwargs) -> bytes:
    """Match the old exact command and fail closed on any other subprocess."""
    if len(argv) != 5 or argv[:2] != ["git", "-C"] or argv[3] != "show":
        raise ValueError("Unrecognized verifier subprocess call")
    if Path(argv[2]).resolve() != relocated_runner.PORT.resolve():
        raise ValueError("Unexpected historical repository path")
    commit, sep, source = argv[4].partition(":")
    if sep != ":" or commit != relocated_runner.PINNED_REGISTRATION or source not in ALLOWED:
        raise ValueError("Unregistered Git source requested by the verifier")
    relocated_runner.frozen_source_check(commit, relocated_runner.HERE / "protocol.json")
    return ALLOWED[source].read_bytes()


def main() -> None:
    # Replace a module attribute, rather than altering subprocess globally.
    verify_followup.subprocess = SimpleNamespace(check_output=historical_git_show)
    verify_followup.main()


if __name__ == "__main__":
    main()
