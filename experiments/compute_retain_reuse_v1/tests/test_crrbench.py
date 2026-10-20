"""Fast CPU sanity tests: python -m pytest tests -q   (or python tests/test_crrbench.py)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from crrbench.tasks import make_task
from crrbench.metrics import first_correct, compute_retain_summary
from crrbench.evalctl import matched_fixed_curve, ece, stop_stats
from crrbench.stats import holm, signflip_p


def test_tasks_fold_consistent():
    rng = np.random.default_rng(0)
    for name in ("cyclic", "dfa", "storyworld", "s3"):
        t = make_task(name)
        x, y, ctx = t.sample(rng, 64, 9)
        assert x.shape == (64, 9 + t.prefix) and x.max() < t.vocab and y.max() < t.n_classes
        # extension composes: fold(N+s) == extend(fold(N), suffix)
        suf = t.sample_suffix(rng, 64, 3)
        sym = x[:, t.prefix:]
        if t.random_init:
            s0 = x[:, 1] - (t.n_symbols + t.n_ctx)
        else:
            s0 = t.init[ctx].copy()
        full = t.fold(ctx, s0.copy(), np.concatenate([sym, suf], 1))
        assert (full == t.extend(ctx, y, suf)).all()


def test_cyclic_is_modular_sum():
    t = make_task("cyclic")
    x, y, _ = t.sample(np.random.default_rng(1), 100, 7)
    assert ((x.sum(1)) % 5 == y).all()


def test_metrics():
    c = np.array([[0, 0, 1, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], bool)
    assert first_correct(c).tolist() == [3, 1, 6]
    s = compute_retain_summary(c, depths=(1, 2, 4))
    assert abs(s["solved_frac"] - 2 / 3) < 1e-9 and abs(s["stable_solve_frac"] - 1 / 3) < 1e-9
    assert abs(s["lost_given_solved"] - 0.5) < 1e-9


def test_matched_fixed_interpolation():
    c = np.zeros((4, 8), bool); c[:, 3:] = True      # correct from loop 4
    assert matched_fixed_curve(c, 3.0).mean() == 0.0 and matched_fixed_curve(c, 4.0).mean() == 1.0
    assert abs(matched_fixed_curve(c, 3.5).mean() - 0.5) < 1e-9


def test_stop_stats():
    kstar = np.array([2, 2, 9]); tau = np.array([1, 4, 3]); ok = np.array([False, True, False])
    s = stop_stats(tau, ok, kstar, 8)
    assert s["premature_rate"] == 0.5 and s["excess_loops"] == 2.0 and s["futile_loops"] == 3.0


def test_stats():
    assert abs(signflip_p([1, 1, 1, 1, 1]) - 2 / 32) < 1e-12
    h = holm([0.01, 0.04, 0.03])
    assert np.allclose(h, [0.03, 0.06, 0.06])
    e, _ = ece(np.linspace(0, 1, 2000), (np.random.default_rng(0).random(2000) < np.linspace(0, 1, 2000)))
    assert e < 0.06


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
