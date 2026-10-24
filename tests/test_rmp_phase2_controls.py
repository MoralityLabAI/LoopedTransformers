from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from research_gym.interp.dictionaries import DictionaryConfig, TopKDictionary, gaussian_source, rms, site_vectors
from research_gym.interp.phase2 import adjacent_width
from research_gym.scripts.rmp_phase2_association import binary_tests, summarize
from research_gym.scripts.rmp_phase2_reported import best_cosines, visit_identity
from research_gym.scripts.rmp_verify_rebuild import verify


def test_adjacent_width_steps_up_except_at_the_top() -> None:
    assert adjacent_width(1024) == 2048
    assert adjacent_width(4096) == 8192
    assert adjacent_width(8192) == 4096


def test_centered_vectors_have_zero_mean_at_every_visit() -> None:
    states = torch.randn(6, 4, 5, 8) + torch.arange(4.0)[None, :, None, None]  # (B, sites, positions, H)
    normalized = states[:, 1:] / rms(states[:, 1:])
    centering = normalized.mean(dim=(0, 2))
    config = DictionaryConfig(kind="centered", width=16, k=2)
    x, y = site_vectors(states, config, layers=3, centering=centering)
    assert x.shape == (6 * 3 * 5, 8) and torch.equal(x, y)
    per_visit = x.reshape(6, 3, 5, 8).mean(dim=(0, 2))
    torch.testing.assert_close(per_visit, torch.zeros(3, 8), atol=1e-5, rtol=0)


def test_gauss_vectors_read_real_states_like_shared_ones() -> None:
    states = torch.randn(2, 4, 3, 8)
    shared = site_vectors(states, DictionaryConfig(kind="shared", width=16, k=2), layers=3)
    gauss = site_vectors(states, DictionaryConfig(kind="gauss", width=16, k=2), layers=3)
    torch.testing.assert_close(shared[0], gauss[0])


def test_gaussian_source_matches_the_requested_moments() -> None:
    generator = torch.Generator().manual_seed(0)
    mixing = torch.randn(4, 4, generator=generator)
    covariance, mean = mixing @ mixing.T + 0.1 * torch.eye(4), torch.tensor([1.0, -2.0, 0.5, 0.0])
    x, y = next(gaussian_source(mean, covariance, 200_000, seed=3))
    assert torch.equal(x, y)
    torch.testing.assert_close(x.mean(dim=0), mean, atol=0.03, rtol=0)
    torch.testing.assert_close(torch.cov(x.T), covariance, atol=0.05, rtol=0.03)


def test_binary_tests_splits_categories_and_drops_thin_classes() -> None:
    select = np.array([True, False] * 50)
    binary = np.array([0, 1] * 25 + [1, 0] * 25)
    tests = binary_tests(binary, select, min_class=10)
    assert [value for value, _ in tests] == [None]
    categorical = np.array([0] * 40 + [1] * 40 + [2] * 16 + [-1] * 4)
    tests = binary_tests(categorical, select, min_class=10)
    assert [value for value, _ in tests] == [0, 1]  # class 2 has only 8 rows in each half
    assert binary_tests(np.full(100, -1), select, min_class=1) == []


def test_summary_counts_found_tests_by_step() -> None:
    tests = [{"family": "f", "variable": "v", "step": step, "found": found, "site": site, "report_auroc": auroc}
             for step, found, site, auroc in [(1, True, 2, 0.9), (1, False, 1, 0.55), (2, True, 3, 0.1)]]
    summary = summarize(tests)["f/v"]
    assert summary["found"] == 2 and summary["tests"] == 3
    assert summary["found_by_step"] == {"1": 1, "2": 1}
    assert summary["found_sites_by_step"] == {"1": [2], "2": [3]}


def test_visit_identity_separates_visit_features_from_input_features() -> None:
    dim, visits, positions = 4, 3, 50
    dictionary = TopKDictionary(dim, 2, 2)
    with torch.no_grad():
        dictionary.pre_bias.zero_()
        dictionary.encoder.copy_(torch.eye(dim)[:, :2])
        dictionary.encoder_bias.zero_()
    states = torch.zeros(4, visits, positions, dim)
    states[:, 0, :, 0] = 5.0  # feature 0 fires only at the first visit
    states[..., 1] = torch.rand(4, visits, positions, generator=torch.Generator().manual_seed(0)) * 3
    share, alive = visit_identity(dictionary, states)  # feature 1 varies the same way at every visit
    assert bool(alive.all())
    assert share[0] > 0.99 and share[1] < 0.05


def test_best_cosines_takes_the_closest_row() -> None:
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[0.6, 0.8], [-1.0, 0.0]])
    torch.testing.assert_close(best_cosines(a, b), torch.tensor([0.6, 0.8]))


def test_verify_rebuild_accepts_identical_records_and_rejects_altered_ones(tmp_path) -> None:
    artifacts = tmp_path / "artifacts" / "cell" / "checkpoints"
    artifacts.mkdir(parents=True)
    (artifacts / "C0.pt").write_bytes(b"weights")
    record = {"cell_hash": "h", "status": "budget_exhausted", "optimizer_steps": 10,
              "checkpoints": [{"label": "C0", "step": 0, "path": "C0.pt",
                               "sha256": hashlib.sha256(b"weights").hexdigest()}]}
    for root in ("committed", "rebuilt"):
        (tmp_path / root / "cell").mkdir(parents=True)
        (tmp_path / root / "cell" / "result.json").write_text(json.dumps(record))
    assert verify("cell", tmp_path / "committed", tmp_path / "rebuilt", tmp_path / "artifacts")["ok"]
    (artifacts / "C0.pt").write_bytes(b"other")
    outcome = verify("cell", tmp_path / "committed", tmp_path / "rebuilt", tmp_path / "artifacts")
    assert not outcome["ok"] and "altered" in outcome["problems"][0]


def test_site_dictionaries_trained_together_equal_ones_trained_alone() -> None:
    from research_gym.interp.analysis import AnalysisModel
    from research_gym.interp.dictionaries import batch_source, train_dictionary, train_site_dictionaries
    from research_gym.interp.models import build_model
    from research_gym.interp.training import CellConfig

    torch.manual_seed(0)
    cell = CellConfig(arm="LOOP-T", seed=0, families=(), hidden_size=16, num_heads=2, visits=3, beta=1.0)
    network = build_model("LOOP-T", hidden_size=16, num_heads=2, visits=3, beta=1.0, vocab_size=32).eval()
    model = AnalysisModel("tiny", "tiny", "C0", 0, cell, network)
    tokens = torch.randint(0, 32, (64, 10), generator=torch.Generator().manual_seed(1))
    for kind, sites in (("trans", [0, 1, 2]), ("visit", [1, 2, 3])):
        configs = [DictionaryConfig(kind=kind, width=32, k=4, site=site, steps=30, batch_examples=4, dead_after=5)
                   for site in sites]
        together = train_site_dictionaries(model, tokens, configs, generator=torch.Generator().manual_seed(0))
        for config, (dictionary, summary) in zip(configs, together):
            source = batch_source(model, tokens, config, generator=torch.Generator().manual_seed(0))
            alone, alone_summary = train_dictionary(source, config, dim=16, device=torch.device("cpu"))
            for name, value in alone.state_dict().items():
                assert torch.equal(dictionary.state_dict()[name], value), (kind, config.site, name)
            assert summary == alone_summary
