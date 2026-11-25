import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lbf19_learned_transfer as t  # noqa: E402

FABLE = (("abc", "def"), ("ab", "de"), ("bf", "cc"))


def test_splits_disjoint_by_canonical_lineage():
    s = t.make_splits(30, 10, 10)
    keys = {n: {t.canonical_lineage(r) for r in v} for n, v in s.items()}
    assert len(keys["train"]) == 30 and len(keys["val"]) == 10 and len(keys["test"]) == 10
    assert not (keys["train"] & keys["val"] or keys["train"] & keys["test"] or keys["val"] & keys["test"])
    # deterministic
    assert t.make_splits(30, 10, 10) == s


def test_canonical_lineage_invariances():
    r = (("ab", "cd"), ("bcd", "efa"))
    perm = {c: "fedcba"[i] for i, c in enumerate(t.ALPHABET)}
    r2 = tuple((("".join(perm[c] for c in v)), ("".join(perm[c] for c in u))) for u, v in reversed(r))
    assert t.canonical_lineage(r) == t.canonical_lineage(r2)


def test_exact_target_matches_repo_transfer_and_concrete_soundness():
    w = "abcabcabfa"
    Y = t.exact_support(FABLE, w, 3)
    assert Y.shape == (4, len(w), 6)
    # K=0 is the word
    assert all(Y[0][p, t.C2I[c]] for p, c in enumerate(w)) and Y[0].sum() == len(w)
    # monotone in k
    assert all((Y[k] <= Y[k + 1]).all() for k in range(3))
    # sound w.r.t. the concrete 3-ball (no false negatives)
    ball, trunc = t.m18.general_concrete_k_ball(FABLE, w, 3)
    for u in ball:
        for p, c in enumerate(u):
            assert Y[3][p, t.C2I[c]]


def test_hull_contains_exact_support_and_gate_is_boolean_step():
    rng = random.Random(5)
    for _ in range(8):
        R = t.sample_system(rng)
        L = rng.choice((8, 11))
        w = t.sample_word(rng, R, L)
        Y = t.exact_support(R, w, 4)
        bt = t.to_batch([(0, R, w, Y)])
        U = t.hull_sequence(bt.w0, bt.Rl, bt.Rr, bt.rm, 4)[:, 0].numpy() > 0.5
        assert (Y <= U).all()


def test_models_shapes_params_and_zero_preservation():
    for a in t.LEARNED_ARMS:
        m = t.make_arm(a)
        assert 20_000 <= m.n_params() <= 80_000
    assert abs(t.make_arm("nonrec_param").n_params() - t.make_arm("rec_prior").n_params()) < 1500
    rng = random.Random(1)
    its = []
    for i in range(4):
        R = t.sample_system(rng)
        w = t.sample_word(rng, R, 9)
        its.append((i, R, w, t.exact_support(R, w, 2)))
    bt = t.to_batch(its)
    torch.manual_seed(0)
    out = t.make_arm("rec_prior")(bt, 5)
    assert out.shape == (5, 4, 9, 6)
    U = t.hull_sequence(bt.w0, bt.Rl, bt.Rr, bt.rm, 5)[1:]
    # support-gated prior: strict support never leaves the exact Boolean hull
    assert ((out > 0) & (U < 0.5)).sum() == 0
    # hard-sound in-loop mask for the unprior arm too
    out2 = t.make_arm("rec_noprior_hs")(bt, 5)
    assert ((out2 > 0) & (U < 0.5)).sum() == 0
    # nonrec supports arbitrary k scalar (extrapolation inputs)
    assert t.make_arm("nonrec_compute")(bt, 8).shape == (8, 4, 9, 6)


def test_metrics_ap_tau_and_ci():
    s = np.array([0.9, 0.8, 0.3, 0.1]); l = np.array([1, 0, 1, 0])
    assert abs(t.average_precision(s, l) - (1 * 0.5 + 0.5 * (2 / 3))) < 1e-9
    assert t.calibrate_tau(np.arange(1, 101, dtype=float)) == 2.0
    m, lo, hi = t.boot_ci(np.array([1.0, 1.0, 1.0, 1.0]))
    assert m == lo == hi == 1.0


def test_quick_end_to_end(tmp_path):
    res = t.run_experiment(tmp_path, quick=True, log=lambda *_: None)
    summ = t.summarize(res, tmp_path)
    assert (tmp_path / "fig_extrapolation_K.svg").read_text().startswith("<svg")
    assert all(v == 0 for v in res["hull_viol"].values())
    # hard-sound hull: structurally zero false negatives
    for sn, d in summ["subsets"].items():
        assert d["pos1_hull"]["recall"]["mean"] == 1.0
