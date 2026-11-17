import random
import pytest

from .gym import Dialogue, Target, run_episode


def test_deterministic_worlds_and_no_false_witness():
    worlds = [Target.sample(random.Random(i)) for i in range(100)]
    for world in worlds:
        d = Dialogue(world)
        for f in range(len(world.bases)):
            for c in range(4):
                d.observe(f, c)
        assert (d.witness() is not None) == world.is_inconsistent()
        if d.witness():
            w = d.witness()
            assert w.first.family == w.second.family
            assert w.first.context in (0, 1, 2)
            assert w.second.context in (0, 1, 2)
            assert w.first.answer != w.second.answer


def test_scoped_exception_excluded():
    world = Target(features=((1, 0), (0, 1)), bases=(0, 1), persona=3)
    d = Dialogue(world)
    d.observe(0, 0)
    d.observe(0, 3)
    assert d.witness() is None
    assert not world.is_inconsistent()


def test_revision_replaces_active_commitment():
    world = Target(features=((1, 0), (0, 1)), bases=(0, 1), persona=1)
    d = Dialogue(world)
    d.observe(0, 0)
    d.observe(0, 1)
    assert d.witness() is not None
    d.revise(0, 1, world.answer(0, 0))
    assert d.witness() is None
    assert len(d.history) == 3


def test_baselines_and_oracle():
    for i in range(40):
        target = Target.sample(random.Random(i))
        for policy in ("random", "paired", "oracle"):
            result = run_episode(target, policy, budget=12, rng=random.Random(i))
            assert result["queries"] <= 12
            assert not result["found"] or result["ground_truth"]
            assert bool(result["witness"]) == result["found"]
            if result["witness"]:
                first, second = (result["witness"][key] for key in ("first", "second"))
                assert first["family"] == second["family"]
                assert first["answer"] != second["answer"]


def test_no_latent_leak_in_dialogue_observations():
    target = Target(features=((1, 0), (0, 1)), bases=(0, 1), persona=4)
    d = Dialogue(target)
    before = d.observations()
    assert before == [[-1] * 4, [-1] * 4]
    d.observe(0, 0)
    after = d.observations()
    assert after[0][0] == target.answer(0, 0)
    assert sum(v != -1 for row in after for v in row) == 1


def test_neural_gradients_and_policy():
    torch = pytest.importorskip("torch")
    from .neural import CounterfactualBeliefField, MirrorPolicy, sample_batch, train
    torch.manual_seed(4)
    model = CounterfactualBeliefField(families=3, width=32, heads=4, loops=2, rank=8)
    feat, obs, labels, risks = sample_batch(random.Random(4), 3, 4, "cpu")
    logits, risk_logits = model(feat, obs)
    assert logits.shape == (4, 3, 4)
    assert risk_logits.shape == (4, 3)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    assert model.shared_block.self_attn.in_proj_weight.grad is not None
    receipt = train(model, steps=2, batch_size=4, seed=4)
    assert receipt["prediction_loss"] > 0
    d = Dialogue(Target.sample(random.Random(11), 3))
    f, c = MirrorPolicy(model).choose(d)
    assert (f, c) in d.available()
    assert c in (0, 1, 2)
