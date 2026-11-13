import torch

from mechinterp.loopscope01.scripts import loopscope_toy as L
from mechinterp.loopscope01.scripts import loopscope_toy_q7 as Q


def tiny():
    torch.manual_seed(0)
    m = L.LoopedChase(n_nodes=8, d=32, layers=1, heads=2).double().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def state(m, B=3, t=3):
    tokens, hops = L.make_batch(8, B, 4, 8)
    e = Q.embed(m, tokens)
    return e, Q.trajectory(m, e, t)[-1], tokens, hops


def test_explicit_transition_matches_forward():
    m = tiny()
    tokens, _ = L.make_batch(8, 5, 1, 8)
    hs = Q.trajectory(m, Q.embed(m, tokens), 4)
    assert torch.allclose(torch.stack([h[:, -1] for h in hs]), m(tokens, 4), atol=1e-10)


def test_jvp_matches_finite_difference():
    m = tiny()
    e, h, _, _ = state(m)
    f = lambda x: Q.step(m, x, e)  # noqa: E731
    v = torch.randn(h.shape, generator=torch.Generator().manual_seed(2), dtype=h.dtype)
    eps = 1e-6
    fd = (f(h + eps * v) - f(h - eps * v)) / (2 * eps)
    assert torch.allclose(Q.jvp(f, (h,), (v,))[1], fd, rtol=1e-5, atol=1e-7)


def test_power_iteration_matches_svd_and_restricted_identity():
    m = tiny()
    e, h, _, _ = state(m, B=2)
    f = lambda x: Q.step(m, x, e)  # noqa: E731
    s_full = Q.sigma_full(f, h, iters=300)
    s_cur = Q.sigma_full(f, h, iters=300, cursor=True)
    I = torch.eye(32, dtype=h.dtype)
    _, s_res = Q.restricted(f, h, I, I)
    for b in range(2):
        J = torch.autograd.functional.jacobian(lambda x: Q.step(m, x[None], e[b:b + 1])[0], h[b])
        J = J.reshape(h[b].numel(), h[b].numel())
        assert abs(float(s_full[b]) / float(torch.linalg.svdvals(J)[0]) - 1) < 1e-6
        Jcc = J.view(*h[b].shape, *h[b].shape)[-1, :, -1, :]
        top = float(torch.linalg.svdvals(Jcc)[0])
        assert abs(float(s_cur[b]) / top - 1) < 1e-6
        assert abs(float(s_res[b]) / top - 1) < 1e-9


def test_analyse_runs_and_shapes():
    m = tiny()
    tokens, hops = L.make_batch(8, 32, 3, 8)
    r = Q.analyse(m, tokens, hops, T=8)
    assert len(r["profile_median"]["pca_rel"]) == 7 and set(r["finite"]) == {"eps0.1", "eps0.3"}


def test_auc():
    import numpy as np
    s = np.array([0.1, 0.2, 0.9, 0.5])
    fails = np.array([False, False, True, False])
    assert Q.auc(s, fails) == 1.0
    assert Q.auc(np.array([1.0, 1.0]), np.array([True, False])) == 0.5
