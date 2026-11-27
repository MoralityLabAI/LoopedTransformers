import torch

from research_gym.interp.molt_carry_feasibility import (
    fresh_roots, halfspaces, legal_gap, projection,
)
from research_gym.interp.molt_amortized_repair import root_pool
from research_gym.interp.molt_clock_repair import donor_world
from research_gym.interp.molt_typed_clock import TypedClockDecoder


def test_dual_obstruction_and_feasible_projection():
    a = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    b = torch.tensor([1., 2.], dtype=a.dtype)
    c, r = projection(a, b, 3.)
    torch.testing.assert_close(c, b)
    assert not r["obstruction_witness"]
    assert abs(r["required_dose_lower_bound"] - 5**.5) < 1e-12
    c, r = projection(a, b, 1.)
    assert r["obstruction_witness"]
    assert c.norm() <= 1. + 1e-12
    _, r = projection(torch.zeros_like(a), b, 10.)
    assert r["obstruction_witness"]
    assert r["required_dose_lower_bound"] is None


def test_layernorm_halfspace_sandwich_with_nonzero_bias():
    torch.set_num_threads(1)
    torch.manual_seed(18)
    model = TypedClockDecoder("addressed").double().eval()
    with torch.no_grad():
        model.decoder.final_norm.bias.copy_(torch.randn(64, dtype=torch.float64))
        h = torch.randn(64, dtype=torch.float64)
        q = torch.linalg.qr(torch.randn(64, 8, dtype=h.dtype)).Q
        # Deliberately exercise both bias signs and a feasible winner.
        coeff = torch.randn(32, 8, dtype=h.dtype)
        coeff = coeff / coeff.norm(dim=-1, keepdim=True) * 2
        for winner in range(17):
            a, n, s, (lo, hi) = halfspaces(model, h, q, 2., winner)
            assert lo <= hi
            assert bool((n <= s + 1e-10).all())
            for c in coeff:
                t = h + q @ c
                denom = ((t - t.mean()).square().mean() + model.decoder.final_norm.eps).sqrt()
                assert lo - 1e-10 <= float(denom) <= hi + 1e-10
                gap = legal_gap(model, t, winner)
                if gap >= 0:
                    assert float((a @ c - n).min()) >= -1e-9
                if float((a @ c - s).min()) >= 0:
                    assert gap >= -1e-9
            winner_found = any(legal_gap(model, h + q @ c, winner) > 1e-8 for c in coeff)
            _, receipt = projection(a, n, 2.)
            if winner_found:
                assert not receipt["obstruction_witness"]


def test_roster_disjoint_from_controller_training_and_development():
    a, b = root_pool(4, 64, 32)
    c = fresh_roots(4, 16)
    def group(roots):
        return {x for w in roots for x in (w, donor_world(w, "changed"), donor_world(w, "same"))}
    assert len(group(c)) == 48
    assert not group(c) & group(a + b)
