import torch

from mechinterp.loopscope01.scripts import loopscope_toy as L
from mechinterp.loopscope01.scripts import loopscope_toy_q11 as Q11


def test_sigma_decode_and_advance():
    tok, hops = L.make_batch(16, 64, 3, 16)
    sig = Q11.sigma_of(tok)
    assert torch.equal(torch.gather(sig, 1, hops[:, :-1]), hops[:, 1:])
    z = torch.nn.functional.one_hot(hops[:, 0], 16).float()
    assert torch.equal(Q11.advance(z, sig).argmax(-1), hops[:, 1])
    z = torch.softmax(torch.randn(64, 16), -1)
    assert torch.allclose(Q11.advance(z, sig).norm(dim=-1), z.norm(dim=-1))  # isometry


def test_cycle_len():
    hops = torch.tensor([[1, 2, 1, 2, 1, 2], [3, 3, 3, 3, 3, 3], [1, 2, 3, 4, 5, 6]])
    assert Q11.cycle_len(hops).tolist() == [2, 1, 6]


def test_perfect_chaser_has_zero_rho_e_hat_and_phi():
    tok, hops = L.make_batch(16, 512, 4, 16)
    sig = Q11.sigma_of(tok)
    z = torch.nn.functional.one_hot(hops.T, 16).float()  # [16, n, 16] perfect beliefs
    rho = Q11.rho_curve(z, sig)
    assert float(rho.max()) == 0.0
    P, _ = Q11.e_hat(rho, Q11.margin(z[7]))
    assert float(P.mean()) == 0.0
    ph = Q11.phase_obstruction(z, Q11.cycle_len(hops))
    assert ph["near_fiber"] > 0 and ph["phi"] in (None, 0.0)


def test_phase_obstruction_detects_phase_dependent_map():
    n = 300
    cyc = torch.full((n,), 2)
    pi = torch.tensor([0, 1, 0, 2, 0, 1, 0, 1])  # node 0 at loops 1 and 3, different successors
    z = torch.nn.functional.one_hot(pi, 16).float()[:, None, :].repeat(1, n, 1)
    ph = Q11.phase_obstruction(z, cyc)
    assert ph["phi"] is not None and ph["phi"] > 0
