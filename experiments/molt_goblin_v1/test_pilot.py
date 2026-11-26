import pytest

torch = pytest.importorskip("torch")

from experiments.molt_goblin_v1.pilot import PrefixReasoner, batch


def test_prefix_stabilization_and_sequential_parity():
    torch.manual_seed(7)
    m = PrefixReasoner(width=8)
    x, _ = batch(5, 2, 6, "cpu")
    with torch.no_grad():
        fixed = m.sequential(x)
        torch.testing.assert_close(m.states(x, 6, "prefix_kv"), fixed, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(m.states(x, 7, "prefix_kv"), fixed, atol=1e-5, rtol=1e-5)


def test_future_tokens_cannot_influence_prefix():
    torch.manual_seed(3)
    m = PrefixReasoner(width=8)
    x, _ = batch(11, 2, 6, "cpu")
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 7
    for arm in ("prefix_kv", "residual", "one_pass"):
        torch.testing.assert_close(m.states(x, 4, arm)[:, :4], m.states(changed, 4, arm)[:, :4])


def test_trainable_and_finite():
    m = PrefixReasoner(width=8)
    x, y = batch(21, 2, 6, "cpu")
    loss = torch.nn.functional.cross_entropy(m(x, 3, "prefix_kv").flatten(0, 1), y.flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert m.block.q.weight.grad is not None
    assert torch.isfinite(m.block.q.weight.grad).all()
