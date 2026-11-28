import torch

from research_gym.interp.aux_gym_v2_observable_bpd import (
    gate, native_span, observer, run_write, trajectory,
)
from research_gym.interp.models import build_model


def test_native_chart_and_reached_write():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = build_model("LOOP-T", hidden_size=16, num_heads=2, visits=4, vocab_size=32).double().eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    u = torch.randn(4, 16, dtype=torch.float64)
    q = native_span(u)
    d = torch.randn(3, 4, dtype=torch.float64) @ u
    torch.testing.assert_close(d @ q @ q.T, d, atol=1e-10, rtol=0)
    with torch.no_grad():
        clean, captured = run_write(model, tokens, 2, capture=True)
        zero, _ = run_write(model, tokens, 2, torch.zeros(1, 16, dtype=torch.float64))
        assert captured.shape == (1, 5, 64)
        torch.testing.assert_close(clean.logits, zero.logits, atol=1e-12, rtol=0)
        patched, _ = run_write(model, tokens, 2, d[:1])
        torch.testing.assert_close(patched.visit_outputs[0], clean.visit_outputs[0])
        torch.testing.assert_close(patched.visit_outputs[1][:, :-1], clean.visit_outputs[1][:, :-1])
        torch.testing.assert_close(patched.visit_outputs[1][:, -1] - clean.visit_outputs[1][:, -1],
                                   model.beta * d[:1], atol=1e-10, rtol=0)
        o, error = observer(model, tokens, 2, q)
        assert o.shape == (3, 32, 4)
        assert error < 0.005
    direct = torch.autograd.functional.jacobian(
        lambda c: trajectory(model, run_write(model, tokens, 2, (c @ q.T)[None])[0], 2)[0],
        torch.zeros(4, dtype=torch.float64),
    )
    torch.testing.assert_close(o, direct, atol=2e-6, rtol=1e-4)


def test_replay_gate_checks_full_trajectory():
    base = torch.zeros(2, 3)
    full = torch.tensor([[1., 0., -1.], [0., 1., -1.]])
    assert gate(full, full.clone(), base)["accepted"]
    changed = full.clone()
    changed[0] = torch.tensor([0., 1., -1.])
    assert not gate(full, changed, base)["accepted"]
    assert not gate(full, 1.5 * full, base)["accepted"]
