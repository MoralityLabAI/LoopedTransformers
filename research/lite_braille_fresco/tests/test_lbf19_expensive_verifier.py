"""Unit tests for LBF-19 expensive-verifier proof search, scaled domain, fast controller, and native Lean."""
from __future__ import annotations

import pathlib
import random
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import lbf19_scaled_domain as sd
import lbf19_native_verifier as nv


def _small_case():
    rules = (("ab", "cd"), ("cd", "ef"), ("ef", "ba"))
    return sd.ScaledCase(
        root="test_small",
        lineage=sd.lineage(rules),
        split="dev",
        family="equality",
        source="ababab",
        target="efcdab",
        rules=rules,
        walk_steps=3,
    )


def test_scaled_bidir_and_replay():
    c = _small_case()
    r = sd.classical_bidir(c, depth=12)
    assert r["closed"]
    assert r["path_len"] == 3
    seen = sd.replay_path(c, r["path"])
    assert seen[0] == c.source and seen[-1] == c.target


def test_lineage_invariance_under_permutation_and_flip():
    r1 = (("ab", "cd"), ("ef", "ac"))
    r2 = (("bd", "fe"), ("dc", "ba"))  # flip + rename a->b,b->a,c->d,d->c,e->f,f->e
    assert sd.lineage(r1) == sd.lineage(r2)


def test_tactic_for_ascii_and_count_mode_dedup():
    c = _small_case()
    tac0 = nv.tactic_for(0, c.source, 0, 0, 0, c.rules)
    tac1 = nv.tactic_for(1, c.target, 1, 1, 2, c.rules)
    assert tac0.isascii() and tac1.isascii()
    assert "lhs" in tac0 and "rhs" in tac1 and "<- " in tac1
    sess = nv.NativeSession(c, repl=None, mode="count")
    sess.discover(0, c.source, (0, 0, 0), "cdabab")
    sess.discover(0, c.source, (0, 0, 0), "cdabab")  # duplicate -> cache hit
    s = sess.summary()
    assert s["edges_requested"] == 2
    assert s["verifier_calls"] == 1
    assert s["cache_hits"] == 1


@pytest.mark.skipif(not (nv.LGYM_EXE.exists() and nv.LEAN_EXE.exists()), reason="lgym/lean not available")
def test_native_lean_correspondence_and_zero_axiom_cert(tmp_path):
    c = _small_case()
    r = sd.classical_bidir(c, depth=12)
    assert r["closed"]
    with nv.start_repl() as repl:
        sess = nv.NativeSession(c, repl, mode="lean", check_goals=True)
        # Replay every step through live NativeSession.discover (checks goal equality at each step!)
        cur = c.source
        for mv in r["path"]:
            sess.discover(0, cur, mv, mv["successor"])
            cur = mv["successor"]
        s = sess.summary()
        assert s["verifier_calls"] == len(r["path"])
        assert s["mismatches"] == 0
        assert s["tactic_seconds"] > 0.0
        cert = sess.certify(r["path"], out_dir=tmp_path, name="test_small_cert")
        assert cert["kernel_ok"] and cert["kernel_axioms"] == []
        assert cert["standalone_ok"]
        batch = nv.compile_batch(c, {"b0": cert["tape"]}, tmp_path, "batch_test")
        assert batch["all_axiom_free"]
