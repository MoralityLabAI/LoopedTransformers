import torch

from research_gym.interp.molt_clock_repair import (
    donor_world, evaluate_case, finite_carry_sham, margin_jacobian, margins, solve,
)
from research_gym.interp.molt_typed_clock import TypedClockDecoder, trace_of


def test_edits_are_oracle_checked():
    world = (3, 1, 19, 2, 16, 4, 19, 7, 18)
    for kind in ("same", "changed"):
        donor = donor_world(world, kind)
        assert donor[5:] == world[5:]
        assert (trace_of(donor)[3] == trace_of(world)[3]) == (kind == "same")


def test_derivative_and_finite_carry_fiber():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    model = TypedClockDecoder("addressed").double().eval()
    h, d = torch.randn(2, 64, dtype=torch.float64)
    with torch.no_grad():
        j = margin_jacobian(model, h)
        exact = torch.func.jacfwd(lambda z: margins(model, z))(h)
        torch.testing.assert_close(j, exact, atol=1e-10, rtol=0)
        s, audit = finite_carry_sham(model, h, d)
        assert audit["status"] == "matched"
        assert audit["direction_distance"] > 1e-5
        assert audit["dose_error"] < 1e-9
        torch.testing.assert_close(margins(model, h + s), margins(model, h + d), atol=1e-9, rtol=0)
        _, impossible = finite_carry_sham(model, h, torch.zeros_like(h))
        assert impossible["status"] == "no_distinct_match"


def test_solve_and_full_native_forward():
    torch.set_num_threads(1)
    torch.manual_seed(37)
    model = TypedClockDecoder("addressed").double().eval()
    h = torch.randn(64, dtype=torch.float64)
    basis = torch.linalg.qr(torch.randn(64, 8, dtype=h.dtype)).Q
    with torch.no_grad():
        target = margins(model, h + basis @ torch.full((8,), .1, dtype=h.dtype))
        d, receipt = solve(model, h, target, basis, 1.)
        assert d.norm() <= 1. + 1e-9
        hist = receipt["margin_objective_history"]
        assert len(hist) == 9
        assert hist[-1] < hist[0]
        assert all(a >= b for a, b in zip(hist, hist[1:]))
        v = torch.randn(256, 8, dtype=h.dtype) * .02
        u = torch.randn(8, 64, dtype=h.dtype) * .02
        world = (3, 1, 19, 2, 16, 4, 19, 7, 18)
        row = evaluate_case(model, world, donor_world(world, "changed"), v, u,
                            torch.linalg.qr(u.T).Q, 41)
        assert len(row["arms"]) == 7
        assert row["sham"]["future_logit_error"] < 1e-9
        assert row["arms"][-1]["margin_error_l2"] < 1e-9
