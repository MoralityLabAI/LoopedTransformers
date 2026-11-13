import torch

from mechinterp.loopscope01.scripts import loopscope_toy as L
from mechinterp.loopscope01.scripts import loopscope_toy_q7 as Q
from mechinterp.loopscope01.scripts import loopscope_toy_q9 as Q9


def tiny():
    torch.manual_seed(0)
    m = L.LoopedChase(n_nodes=8, d=32, layers=1, heads=2).double().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def test_state_memory_matches_autograd_grad():
    m = tiny()
    tokens, _ = L.make_batch(8, 4, 1, 8)
    r = Q9.state_memory(m, tokens, T=4)
    e = Q.embed(m, tokens)
    h1 = Q.step(m, torch.zeros_like(e), e).detach().requires_grad_(True)
    h = h1
    for _ in range(3):
        h = Q.step(m, h, e)
    h4 = h
    y = Q9.own_margin(m.readout(h4[:, -1])).sum()
    g1, g4 = torch.autograd.grad(y, (h1, h4))
    n = lambda x: x.flatten(1).norm(dim=1)  # noqa: E731
    ref = n(g1) * n(h1) / (n(g4) * n(h4))
    assert torch.allclose(torch.as_tensor(r), ref.detach(), rtol=1e-8)


def test_clock_probe_and_subspace_on_planted_clock():
    g = torch.Generator().manual_seed(0)
    d = 16
    offs = torch.randn(9, d, generator=g) * 3
    mk = lambda n: [Q9.unit(torch.randn(n, d, generator=g) * 0.1 + offs[t]) for t in range(8)]  # noqa: E731
    U_tr, U_te = mk(64), mk(32)
    assert Q9.clock_probe(U_tr, U_te) > 0.95
    mu, P = Q9.clock_subspace(U_tr)
    assert P.shape == (d, 6) and torch.allclose(P.T @ P, torch.eye(6), atol=1e-5)


def test_run_ablated_clean_matches_forward_and_ablation_removes_component():
    m = tiny()
    tokens, hops = L.make_batch(8, 16, 2, 8)
    clean = Q9.run_ablated(m, tokens, 5, None, None)
    with torch.no_grad():
        ref = m.readout(m(tokens, 5)).argmax(-1)
    assert torch.equal(clean, ref.cpu())
    P = torch.linalg.qr(torch.randn(32, 3, dtype=torch.float64))[0]
    mu = torch.zeros(32, dtype=torch.float64)
    u = Q9.unit(torch.randn(5, 32, dtype=torch.float64))
    ua = u - ((u - mu) @ P) @ P.T
    assert torch.allclose(ua @ P, torch.zeros(5, 3, dtype=torch.float64), atol=1e-12)


def test_lock_in():
    assert Q9.lock_in([0.1, 0.95, 0.2, 0.9, 1, 1, 1, 1, 0.1]) == 4
    assert Q9.lock_in([1.0] * 8) == 1
