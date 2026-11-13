import torch

from mechinterp.loopscope01.scripts import loopscope_toy as L
from mechinterp.loopscope01.scripts import loopscope_toy_q8 as Q8


def test_residual_fraction_zero_inside_one_orthogonal():
    P = torch.eye(5)[:, :2]
    mu = torch.zeros(5)
    U = torch.tensor([[1.0, 2.0, 0, 0, 0], [0, 0, 3.0, 0, 0], [1.0, 0, 1.0, 0, 0]])
    r = Q8.residual_fraction(U, mu, P)
    assert torch.allclose(r, torch.tensor([0.0, 1.0, 0.5]))


def test_ref_subspace_captures_planted_plane():
    g = torch.Generator().manual_seed(0)
    basis = torch.linalg.qr(torch.randn(10, 2, generator=g, dtype=torch.float64))[0]
    U = torch.randn(200, 2, generator=g, dtype=torch.float64) @ basis.T + 3.0
    mu, P = Q8.ref_subspace(U, k=2)
    assert float(Q8.residual_fraction(U, mu, P).max()) < 1e-8


def test_analyse_runs_on_tiny_model():
    torch.manual_seed(0)
    m = L.LoopedChase(n_nodes=8, d=32, layers=1, heads=2).double().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    tr, _ = L.make_batch(8, 32, 1, 16)
    td, hd = L.make_batch(8, 32, 2, 16)
    r = Q8.analyse(m, tr, td, hd)
    assert len(r["r_median_by_loop"]) == 16 and len(r["gamma_median_by_transition"]) == 15
    assert len(r["acc_by_loop"]) == 16
