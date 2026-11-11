from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from research_gym.interp.analysis import AnalysisModel
from research_gym.interp.dictionaries import (
    DictionaryConfig,
    TopKDictionary,
    fraction_of_variance_unexplained,
    model_states,
    site_vectors,
    train_dictionary,
)
from research_gym.interp.models import build_model
from research_gym.interp.replacement import (
    answer_scores,
    at_visits,
    ce_recovered,
    fit_pca,
    matched_noise,
    pca_components_matching,
    pca_truncation,
    reconstruct,
    transcoder_step,
)
from research_gym.interp.training import CellConfig


CPU = torch.device("cpu")


def _sparse_source(dim: int, atoms: int, active: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    dictionary = torch.randn(atoms, dim, generator=generator)
    dictionary /= dictionary.norm(dim=1, keepdim=True)
    while True:
        codes = torch.zeros(1024, atoms)
        index = torch.stack([torch.randperm(atoms, generator=generator)[:active] for _ in range(1024)])
        codes.scatter_(1, index, torch.rand(1024, active, generator=generator) + 0.5)
        x = codes @ dictionary
        yield x, x


def test_topk_autoencoder_recovers_sparse_data_and_keeps_unit_decoder_rows() -> None:
    config = DictionaryConfig(kind="shared", width=128, k=4, steps=1500, learning_rate=3e-3)
    source = _sparse_source(16, 48, 3, 0)
    dictionary, summary = train_dictionary(source, config, dim=16, device=CPU)
    x, y = next(source)  # fresh samples from the same true dictionary
    assert fraction_of_variance_unexplained(dictionary, x, y) < 0.15
    torch.testing.assert_close(dictionary.decoder.norm(dim=1), torch.ones(128), atol=1e-5, rtol=0)
    assert (dictionary.encode(x) > 0).sum(dim=-1).max() <= 4
    assert summary["config"]["width"] == 128 and summary["dead_latents"] >= 0


def test_transcoder_learns_a_linear_map_through_its_skip_term() -> None:
    generator = torch.Generator().manual_seed(2)
    matrix = torch.randn(8, 8, generator=generator) / 3

    def source():
        while True:
            x = torch.randn(512, 8, generator=generator)
            yield x, x @ matrix

    config = DictionaryConfig(kind="trans", width=32, k=2, steps=800, learning_rate=1e-2, dead_after=10)
    dictionary, summary = train_dictionary(source(), config, dim=8, device=CPU)
    x, y = next(source())
    assert dictionary.skip is not None
    assert fraction_of_variance_unexplained(dictionary, x, y) < 0.05
    assert summary["train_fvu"] < 0.1


def _tiny_model(arm: str = "LOOP-T", injection: bool = False) -> AnalysisModel:
    torch.manual_seed(0)
    config = CellConfig(arm=arm, seed=0, families=(), hidden_size=16, num_heads=2, visits=3,
                        beta=1.0, input_injection=injection)
    model = build_model(arm, hidden_size=16, num_heads=2, visits=3, beta=1.0, vocab_size=32).eval()
    return AnalysisModel("tiny", "tiny", "C0", 0, config, model)


def test_site_vectors_shapes_and_normalization() -> None:
    model = _tiny_model()
    tokens = torch.randint(0, 32, (4, 6))
    states = model_states(model, tokens)
    assert states.shape == (4, 4, 6, 16)
    shared = DictionaryConfig(kind="shared", width=8, k=2)
    x, y = site_vectors(states, shared, layers=3)
    assert x.shape == (4 * 3 * 6, 16) and torch.equal(x, y)
    torch.testing.assert_close(x.pow(2).mean(dim=-1), torch.ones(x.shape[0]), atol=1e-4, rtol=0)
    trans = DictionaryConfig(kind="trans", width=8, k=2, site=1)
    x, y = site_vectors(states, trans, layers=3)
    scale = states[:, 1].pow(2).mean(dim=-1, keepdim=True).sqrt().reshape(-1, 1)
    torch.testing.assert_close(y * scale, (states[:, 2] - states[:, 1]).reshape(-1, 16), atol=1e-4, rtol=1e-4)


def test_splices_change_only_the_hooked_visit_and_identity_keeps_scores() -> None:
    model = _tiny_model("FF-U", injection=True)
    tokens = torch.randint(0, 32, (5, 6))
    targets = torch.randint(0, 32, (5,))
    clean_loss, clean_ok = answer_scores(model, tokens, targets)
    same_loss, same_ok = answer_scores(model, tokens, targets, at_visits(lambda context: context.after))
    np.testing.assert_allclose(same_loss, clean_loss, rtol=1e-5)

    seen: list[int] = []

    def mark(context):
        seen.append(context.visit)
        return context.after

    answer_scores(model, tokens, targets, at_visits(mark, [1]))
    assert seen == [1]

    class Zero:
        def predict_update(self, state):
            return torch.zeros_like(state)

    # A zero update everywhere leaves the embedding untouched through every visit.
    captured: list[torch.Tensor] = []

    def capture(context):
        out = transcoder_step([Zero()])(context)
        captured.append(out)
        return out

    model.run(tokens, at_visits(capture))
    entering = model.model.embed(tokens)
    torch.testing.assert_close(captured[-1], entering)


def test_noise_matches_reconstruction_error_and_full_pca_is_identity() -> None:
    torch.manual_seed(3)
    dictionary = TopKDictionary(16, 32, 2, seed=1)
    after = torch.randn(4, 6, 16) * 3

    class Context:
        pass

    context = Context()
    context.after = after
    noisy = matched_noise(dictionary, seed=0)(context)
    error = (after - dictionary.reconstruct_state(after)).norm(dim=-1)
    torch.testing.assert_close((noisy - after).norm(dim=-1), error, atol=1e-4, rtol=1e-4)
    assert torch.allclose(reconstruct(dictionary)(context), dictionary.reconstruct_state(after))

    x = torch.randn(2000, 16) * torch.linspace(3, 0.1, 16)
    mean, components, variances = fit_pca(x)
    assert torch.all(variances[:-1] >= variances[1:])
    full = pca_truncation(mean, components)(context)
    torch.testing.assert_close(full, after, atol=1e-4, rtol=1e-4)
    total = float(variances.sum())
    assert pca_components_matching(variances, total + 1, total) == 1
    assert pca_components_matching(variances, 0.0, total) == 16
    assert abs(ce_recovered(1.0, 3.0, 1.5) - 0.75) < 1e-9
