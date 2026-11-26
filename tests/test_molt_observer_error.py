import pytest
import torch

from experiments.molt_goblin_v1.mms_error_study import METHODS, run
from experiments.molt_goblin_v1.observability_gym import (
    DelayChain, analytic, cut_states, observe, sample,
)
from experiments.molt_goblin_v1.observer_error import (
    directional_interval, global_curvature, linear_box,
    spectral_segment_bound, stable_signs, stable_symmetric,
)
from lsa.observable_memory import Sketch


def test_linear_box_and_gate():
    w = torch.tensor([[2., -3.], [-1., 4.]], dtype=torch.float64)
    lo = torch.tensor([-1., 2.], dtype=w.dtype)
    hi = torch.tensor([3., 5.], dtype=w.dtype)
    a, b = linear_box(w, lo, hi)
    torch.testing.assert_close(a, torch.tensor([-17., 5.], dtype=w.dtype))
    torch.testing.assert_close(b, torch.tensor([0., 21.], dtype=w.dtype))
    z = torch.tensor([2., -3.], dtype=w.dtype)
    assert stable_symmetric(z, 1)
    assert not stable_symmetric(z, 2)
    assert stable_signs(z, torch.tensor([1., -8.]), torch.tensor([3., 2.]))
    assert not stable_signs(z, torch.tensor([-3., -8.]), torch.tensor([3., 2.]))
    with pytest.raises(ValueError):
        spectral_segment_bound(-1, 1, 1, 1)


def test_bound_encloses_actual_nonlinear_difference():
    torch.manual_seed(3)
    m = DelayChain().double()
    g = torch.Generator().manual_seed(17)
    x = sample(97, 1, 8, torch.float64)[0]
    p = 2
    _, base = cut_states(m, x, p)
    _, _, cone, _ = observe(m, x, base, p)
    o = analytic(m, x, base, p)
    s = Sketch.build(o, 5)
    hess = global_curvature(m, x, p)
    for k in (1, 3, 5):
        for scale in (0.01, 1.0, 4.0):
            full = scale * torch.randn(16, dtype=x.dtype, generator=g)
            proj = s.project(full, k)
            a = observe(m, x, base, p, proj)[0][cone].detach()
            b = observe(m, x, base, p, full)[0][cone].detach()
            difference = b - a
            bound = spectral_segment_bound(s.tail(k), float((full - proj).norm()),
                                           float(full.norm()), hess)
            assert float(difference.norm()) <= bound + 1e-8
            old_width = None
            for segments in (1, 4, 16):
                lo, hi = directional_interval(m, x, base, p, proj, full, segments)
                assert bool(((lo <= difference) & (difference <= hi)).all())
                if old_width is not None:
                    assert float((hi - lo).sum()) <= old_width + 1e-8
                old_width = float((hi - lo).sum())
                if stable_signs(a, lo, hi):
                    assert bool((a.sign() == b.sign()).all())
    zero = torch.zeros(16, dtype=x.dtype)
    lo, hi = directional_interval(m, x, base, p, zero, zero, 1)
    assert float(hi.abs().max()) <= 1e-9
    assert float(lo.abs().max()) <= 1e-9


def test_smoke_preserves_full_roster():
    report = run(2, 1, (3,))
    assert len(report["rows"]) == 2 * 3 * 5
    assert len(report["decisions"]) == 2 * 3 * len(METHODS)
    assert not report["formal_certificate_authorized"]
    for row in report["summary"]:
        assert row["accepted"] + row["fallback"] == row["rows"]
        if row["method"] != "linear_only":
            assert row["false_accepts"] == 0
            assert row["task_correct"] == row["full_task_correct"]
