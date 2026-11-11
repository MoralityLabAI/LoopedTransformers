from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lsa.kappa_probe import estimate_kappa  # noqa: E402
from research_gym.interp.hooks import (  # noqa: E402
    capture,
    compose,
    patch,
    record_components,
    repeat,
    skip,
    stop_after,
)
from research_gym.interp.models import build_model  # noqa: E402
from research_gym.neural.looped_decoder import DecoderBlock, ExplicitDecoderBlock  # noqa: E402


def _model(arm: str, seed: int = 3):
    torch.manual_seed(seed)
    return build_model(arm, hidden_size=32, num_heads=4, visits=4, vocab_size=64).eval()


def _tokens(seed: int = 5) -> torch.Tensor:
    return torch.randint(0, 64, (3, 12), generator=torch.Generator().manual_seed(seed))


def test_rung_m0_parameter_counts_match_the_program() -> None:
    assert build_model("LOOP-T").parameter_breakdown()["unique_parameters"] == 459_776
    assert build_model("FF-U").parameter_breakdown()["unique_parameters"] == 1_839_616


@pytest.mark.parametrize("arm", ["LOOP-T", "FF-U"])
def test_hooks_that_do_nothing_leave_outputs_bit_identical(arm: str) -> None:
    model, tokens = _model(arm), _tokens()
    plain = model(tokens)
    seen: list[torch.Tensor] = []
    for hook in (
        lambda context: None,
        lambda context: context.after,
        compose(),
        compose(capture(seen), lambda context: None),
    ):
        hooked = model(tokens, visit_hook=hook)
        assert torch.equal(hooked.logits, plain.logits)
        assert all(torch.equal(a, b) for a, b in zip(hooked.visit_outputs, plain.visit_outputs))
    assert len(seen) == 4 and all(torch.equal(a, b) for a, b in zip(seen, plain.visit_outputs))


def test_explicit_block_computes_the_same_function_as_the_torch_block() -> None:
    torch.manual_seed(11)
    reference = DecoderBlock(32, 4, alpha=1.0, beta=0.25, normalization="pre")
    explicit = ExplicitDecoderBlock(32, 4, alpha=1.0, beta=0.25)
    explicit.qkv.weight.data.copy_(reference.attention.in_proj_weight)
    explicit.attention_out.weight.data.copy_(reference.attention.out_proj.weight)
    explicit.mlp.load_state_dict(reference.mlp.state_dict())
    explicit.norm1.load_state_dict(reference.norm1.state_dict())
    explicit.norm2.load_state_dict(reference.norm2.state_dict())
    state = torch.randn(2, 9, 32)

    assert sum(p.numel() for p in explicit.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )
    assert torch.allclose(explicit(state), reference(state), atol=1e-5)


@pytest.mark.parametrize("alpha", [1.0, 0.9])
def test_update_identity_and_head_decomposition_are_exact(alpha: float) -> None:
    torch.manual_seed(13)
    block = ExplicitDecoderBlock(32, 4, alpha=alpha, beta=0.25)
    state = torch.randn(2, 9, 32)
    parts = block.components(state)
    delta = parts["resid_post"] - parts["resid_pre"]
    predicted = (
        (alpha**2 - 1.0) * parts["resid_pre"]
        + alpha * block.beta * parts["attn_out"]
        + block.beta * parts["mlp_out"]
    )

    assert torch.allclose(delta, predicted, atol=1e-5)
    assert torch.allclose(parts["head_out"].sum(dim=1), parts["attn_out"], atol=1e-5)
    assert torch.allclose(parts["pattern"].sum(dim=-1), torch.ones(2, 4, 9), atol=1e-5)
    assert torch.equal(parts["pattern"].triu(diagonal=1), torch.zeros(2, 4, 9, 9))
    assert torch.equal(parts["resid_post"], block(state))


@pytest.mark.parametrize("arm", ["LOOP-T", "FF-U"])
def test_recorded_components_chain_across_visits(arm: str) -> None:
    model, tokens = _model(arm), _tokens()
    plain = model(tokens)
    with record_components(model) as records:
        recorded = model(tokens)

    assert len(records) == 4
    assert torch.equal(records[0]["resid_pre"], model.embed(tokens))
    for earlier, later in zip(records, records[1:]):
        assert torch.equal(earlier["resid_post"], later["resid_pre"])
    assert torch.equal(recorded.logits, plain.logits)
    assert all(block.recorder is None for block in model.blocks)


def test_skip_repeat_and_stop_match_runs_of_other_lengths_in_a_tied_loop() -> None:
    model, tokens = _model("LOOP-T"), _tokens()
    plain = model(tokens)

    assert torch.allclose(
        model(tokens, visit_hook=skip(3)).logits, model(tokens, depth_visits=3).logits, atol=1e-6
    )
    assert torch.allclose(
        model(tokens, visit_hook=repeat(3)).logits, model(tokens, depth_visits=5).logits, atol=1e-6
    )
    stopped = model(tokens, visit_hook=stop_after(1))
    assert len(stopped.visit_outputs) == 2
    assert torch.equal(stopped.logits, model.lens(plain.visit_outputs[1]))


@pytest.mark.parametrize("arm", ["LOOP-T", "FF-U"])
def test_full_state_patch_transplants_the_other_run(arm: str) -> None:
    model = _model(arm)
    base, source = _tokens(5), _tokens(7)
    source_run = model(source)
    patched = model(base, visit_hook=patch(1, source_run.visit_outputs[1]))

    assert not torch.allclose(model(base).logits, source_run.logits)
    assert torch.equal(patched.logits, source_run.logits)
    doubled = model(base, visit_hook=patch(2, lambda state: 2.0 * state))
    assert not torch.allclose(doubled.logits, model(base).logits)


@pytest.mark.parametrize("arm", ["LOOP-T", "FF-U"])
def test_decoder_implements_the_residual_loop_interface(arm: str) -> None:
    model, tokens = _model(arm), _tokens()
    plain = model(tokens)
    states = model.states(model.embed(tokens))

    assert len(states) == model.rounds + 1 == 5
    assert all(torch.equal(a, b) for a, b in zip(states[1:], plain.visit_outputs))
    assert torch.equal(model.suffix(states[2], 1), states[-1])
    assert torch.equal(model.lens(states[-1]), plain.logits)
    assert (model.block_at(0) is model.block_at(3)) == (arm == "LOOP-T")

    inputs = model.embed(tokens).detach()
    estimate = estimate_kappa(model, inputs, torch.zeros_like(inputs), power_iterations=2)
    assert 0.0 <= estimate.kappa <= model.rounds
    assert len(estimate.u_norms) == len(estimate.g_norms) == model.rounds
