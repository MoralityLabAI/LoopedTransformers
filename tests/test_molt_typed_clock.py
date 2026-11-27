import torch

from research_gym.interp.molt_typed_clock import (
    MOD, OPERATIONS, TypedClockDecoder, competent, evaluate, make_worlds,
    primitive_bank, target_step, trace_of,
)


def test_exhaustive_oracle_and_distinct_rosters():
    x, y = primitive_bank()
    assert x.shape == (MOD * OPERATIONS, 2)
    assert len({tuple(r) for r in x.tolist()}) == 561
    for (s, op), target in zip(x.tolist(), y.tolist(), strict=True):
        assert target == target_step(s, op)
        assert 0 <= target < MOD
    for op in range(OPERATIONS):
        assert len({target_step(s, op) for s in range(MOD)}) == MOD
    a = make_worlds(1, 4, 32)
    b = make_worlds(1, 4, 32, a)
    assert len(set(a + b)) == 64
    assert trace_of((3, 1, 19, 2, 16)) == (6, 9, 10, 10)


def test_continuous_carry_native_shared_site():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    m = TypedClockDecoder("addressed")
    x = torch.tensor([[1, 1, 18, 2, 19], [3, 5, 20, 7, 21]])
    calls = []
    h = m.get_submodule("decoder.blocks.0.mlp.2").register_forward_hook(
        lambda _m, _i, out: calls.append(tuple(out.shape)))
    logits, probs = m.rollout(x)
    h.remove()
    assert logits.shape == (2, 4, 2048)
    assert probs.shape == (2, 4, 17)
    assert calls == [(2, 3, 64)] * 4
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, 4))
    # First primitive is independent of any future operation.
    changed = x.clone()
    changed[:, 2:] = 16
    first, _ = m.rollout(changed, passes=1)
    torch.testing.assert_close(first, logits[:, :1])
    logits[:, -1].square().mean().backward()
    assert m.decoder.embedding.weight.grad is not None
    assert m.decoder.blocks[0].mlp[-1].weight.grad.norm() > 0


def test_evaluation_and_gate():
    torch.manual_seed(7)
    model = TypedClockDecoder("separable")
    result = evaluate(model, make_worlds(17, 2, 4))
    assert result["worlds"] == 4
    assert len(result["stage_exact"]) == 4
    row = {"primitive_after_rollout": {"examples": 561, "unrestricted_correct": 561},
           "development": {"4": {"unrestricted_terminal_exact": 1.},
                           "8": {"unrestricted_terminal_exact": 0.75}}}
    assert not competent(row)
    row["development"]["8"]["unrestricted_terminal_exact"] = 0.85
    assert competent(row)
