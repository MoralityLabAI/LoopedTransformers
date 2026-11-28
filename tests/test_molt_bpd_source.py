import subprocess

import pytest

from research_gym.interp.molt_bpd_source import preflight, verify


def test_preflight_and_child_free_recheck(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    source = checkout / "mechinterp" / "bpd"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("VALUE = 1\n")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args])
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    revision = git("rev-parse", "HEAD").decode().strip()
    receipt = tmp_path / "source.json"
    preflight(checkout, receipt, revision)
    with pytest.raises(ValueError):
        preflight(checkout, tmp_path / "wrong.json", "0" * 40)
    with monkeypatch.context() as m:
        m.setattr(subprocess, "check_output", lambda *a, **k: pytest.fail("worker spawned a process"))
        verify(checkout, receipt, revision)
        (source / "injected.py").write_text("VALUE = 2\n")
        with pytest.raises(ValueError):
            verify(checkout, receipt, revision)
        (source / "injected.py").unlink()
        (source / "__init__.py").write_text("VALUE = 3\n")
        with pytest.raises(ValueError):
            verify(checkout, receipt, revision)
    with pytest.raises(ValueError):
        preflight(checkout, tmp_path / "dirty.json", revision)
