import copy

import pytest

torch = pytest.importorskip("torch")

from experiments.molt_goblin_v1.pilot import PrefixReasoner, batch
from experiments.molt_goblin_v1.study import append_only, objective, plan, run


def test_append_only_forward_and_gradient_parity():
    torch.manual_seed(2)
    m = PrefixReasoner(width=8).double()
    ref = copy.deepcopy(m)
    x, _ = batch(5, 2, 5, "cpu")
    actual = append_only(m, x)
    expected = ref.sequential(x)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual, ref.states(x, 5, "prefix_kv"), atol=1e-10, rtol=1e-10)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a, b in zip(m.parameters(), ref.parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-9, rtol=1e-9)


def test_loss_schedule_and_budget():
    m = PrefixReasoner(width=8)
    x, y = batch(7, 2, 4, "cpu")
    for arm in ("prefix_kv", "residual", "one_pass"):
        for schedule in ("last", "all"):
            assert torch.isfinite(objective(m, x, y, arm, 3, schedule))
    assert ("one_pass", 24, "last", "reinvest_forward_calls") in plan(8, 3, ("last", "all"))


def test_study_smoke():
    torch.set_num_threads(1)
    report = run(1, 2, (31,), ("last", "all"), "cpu")
    assert len(report["results"]) == 6
    baseline = next(r for r in report["results"] if r["panel"] == "reinvest_forward_calls")
    assert baseline["forward_block_calls"] == 2
    assert baseline["optimizer_updates"] == 2
    for row in report["results"]:
        if row["arm"] == "prefix_kv":
            assert row["append_only_parity"]["max_abs_vs_unroll"] < 1e-5
    assert not report["promotion_authorized"]
