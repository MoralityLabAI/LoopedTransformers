"""End-to-end checkpoint import against MIRROR-02's real Transformer class."""
import json

import pytest

torch = pytest.importorskip("torch")
from experiments.mirror_epistemic_transport_v0.v2_compositional import torch_components
from experiments.mirror_epistemic_transport_v0.mirror03.contracts import example_contract
from experiments.mirror_epistemic_transport_v0.mirror03.bridge import (
    NeuralPlanner, FixtureTarget, export_v2_checkpoint, run, verify_receipt)


def test_train_v2_load_into_text_contract(tmp_path):
    cls, _, _, train = torch_components()
    torch.manual_seed(41)
    net = cls(width=32, loops=2, transport="reflection")
    receipt = train(net, seed=41, steps=2, batch_size=4, imitation_weight=.8)
    assert receipt["steps"] == 2
    checkpoint = tmp_path / "v2.pt"
    torch.save({"configuration":{"width":32, "loops":2, "transport":"reflection"},
                "state_dict":net.state_dict()}, checkpoint)
    planner = NeuralPlanner(str(checkpoint))
    contract = example_contract()
    answers = {q.id: json.dumps({"stance":"deny" if q.id == "appeal-roles" else "affirm",
                                 "reason":"test"}) for q in contract.questions}
    result = run(contract, FixtureTarget(answers), planner, budget=5)
    assert result["witness"] is not None
    assert verify_receipt(contract, result)
    assert all(e["question_id"] != "appeal-probe" for e in result["events"])


def test_exported_training_checkpoint_loads(tmp_path):
    checkpoint = tmp_path / "exported.pt"
    receipt = export_v2_checkpoint(str(checkpoint), seed=41, steps=2, width=32,
                                  loops=2, transport="reflection")
    assert receipt["training"]["steps"] == 2
    assert checkpoint.exists()
    planner = NeuralPlanner(str(checkpoint))
    from experiments.mirror_epistemic_transport_v0.mirror03.bridge import Ledger
    contract = example_contract()
    assert planner.choose(contract, Ledger(contract)) in {
        "appeal-neutral", "appeal-roles", "equal-neutral", "equal-consequence"}
