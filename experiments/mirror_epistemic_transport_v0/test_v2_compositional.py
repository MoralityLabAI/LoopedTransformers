import random
import pytest
import torch

from experiments.mirror_epistemic_transport_v0 import v2_compositional as m


def test_split_registries_and_disjoint_contexts():
    assert m.TRAIN_CONTEXTS[:4] == m.OOD_CONTEXTS[:4]
    assert all(t not in m.TRAIN_CONTEXTS for t in m.OOD_CONTEXTS[4:8])
    assert set(m.TRAIN_PERSONAS) & set(m.OOD_PERSONAS) == {0}
    assert len(m.World.sample(random.Random(3), "train").features) == 6
    assert len(m.World.sample(random.Random(3), "ood").features) == 9


def test_certified_witness_scope_and_revision():
    world = m.World(((1, 0, 0),), m.OOD_CONTEXTS, (0,), 1, (1,))
    d = m.Dialogue(world)
    d.ask(0, 0)
    d.ask(0, 1)
    assert m.witness(d.view()) is not None
    d.revise(0, 1, 0)
    assert m.witness(d.view()) is None
    # Scope-only evidence is not allowed to certify a contradiction.
    view = m.View(world.features, world.contexts,
                  ((0, -1, -1, -1, -1, -1, -1, -1, 1),))
    assert m.witness(view) is None


def test_oracle_boundary():
    x = ((1, 1, 0), (0, 1, 0))
    a = m.World(x, m.TRAIN_CONTEXTS, (0, 0), 1, (0, 1))
    b = m.World(x, m.TRAIN_CONTEXTS, (1, 1), 4, (1, 0))
    assert m.Dialogue(a).view() == m.Dialogue(b).view()
    assert m.paired(m.Dialogue(a).view()) == m.paired(m.Dialogue(b).view())
    assert m.exact_bayes(m.Dialogue(a).view()) == m.exact_bayes(m.Dialogue(b).view())


def test_no_false_witness_all_answers():
    for i in range(150):
        w = m.World.sample(random.Random(i), "ood")
        d = m.Dialogue(w)
        for f, row in enumerate(w.features):
            for c in range(len(w.contexts) - 1):
                d.ask(f, c)
        assert (m.witness(d.view()) is not None) == w.inconsistent()


def test_exact_group_composition_and_free_ablation():
    Model, _, _, _ = m.torch_components()
    torch.manual_seed(9)
    f = torch.randn(4, 6, 3)
    reflection = Model(width=32, loops=2, transport="reflection")
    for a in range(8):
        for b in range(8):
            aa = torch.tensor([[float((a >> k) & 1) for k in range(3)]] * 4)
            bb = torch.tensor([[float((b >> k) & 1) for k in range(3)]] * 4)
            assert reflection.composition_error(f, aa, bb) < 1e-10
    free = Model(width=32, loops=2, transport="free")
    assert free.composition_error(f, aa, bb) > 1e-8


def test_training_actor_and_heldout_eval():
    Model, Policy, _, train = m.torch_components()
    torch.manual_seed(2)
    model = Model(width=32, loops=2, transport="reflection")
    stats = train(model, seed=2, steps=3, batch_size=6, imitation_weight=.8)
    assert stats["last_loss"] > 0
    view = m.Dialogue(m.World.sample(random.Random(2), "ood")).view()
    assert Policy(model).choose(view) in view.candidates()
    report = m.predictive_compositions(model, seed=5, episodes=5)
    assert report["n_predictions"] == 5 * 9 * 4
    assert 0 <= report["composed_brier"] <= 1


def test_bayes_baseline_valid_under_ood():
    for i in range(50):
        w = m.World.sample(random.Random(i), "ood")
        d = m.Dialogue(w)
        for _ in range(5):
            q = m.exact_bayes(d.view())
            assert q in d.view().candidates()
            d.ask(*q)
            if m.witness(d.view()):
                break
