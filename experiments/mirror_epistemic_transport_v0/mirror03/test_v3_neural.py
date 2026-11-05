"""Neural bridge contract tests; substitute a deterministic tiny torch actor."""
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from experiments.mirror_epistemic_transport_v0.mirror03.contracts import example_contract
from experiments.mirror_epistemic_transport_v0.mirror03.bridge import NeuralPlanner, Ledger


class StubActor(torch.nn.Module):
    def __init__(self, width=8, loops=2, transport="reflection"):
        super().__init__()
        self.proj = torch.nn.Linear(3, 1)

    def forward(self, features, contexts, observed):
        batch, families, _ = features.shape
        steps = contexts.shape[1]
        rank = (self.proj(features).squeeze(-1)[:, :, None]
                + contexts[:, None, :, 0])
        fake = torch.zeros(batch, families, steps)
        return fake, fake[:, :, 0], rank


def test_loaded_actor_cannot_pick_probe_or_missing_cell(monkeypatch, tmp_path):
    def fake_components():
        return StubActor, None, None, None
    monkeypatch.setitem(sys.modules,
        "experiments.mirror_epistemic_transport_v0.v2_compositional",
        SimpleNamespace(torch_components=fake_components))
    checkpoint = tmp_path / "actor.pt"
    model = StubActor()
    torch.save({"configuration":{"width":8, "loops":2, "transport":"reflection"},
                "state_dict":model.state_dict()}, checkpoint)
    chooser = NeuralPlanner(str(checkpoint))
    contract = example_contract()
    ledger = Ledger(contract)
    q = chooser.choose(contract, ledger)
    assert q in {"appeal-neutral", "appeal-roles", "equal-neutral", "equal-consequence"}
    ledger.record("appeal-neutral", '{"stance":"affirm","reason":"initial"}')
    assert chooser.choose(contract, ledger) == "appeal-roles"
