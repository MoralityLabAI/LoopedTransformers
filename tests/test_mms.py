import copy
import torch
from research_gym.interp.mms import GradientBank, MechanisticMemory


def test_cache_budget_collision_age_and_resume():
    b = GradientBank(0.00004, (4,))
    assert b.capacity == 1
    b.write(torch.tensor([0, 1]), torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]]), 0)
    v, hit = b.read(torch.tensor([0, 1]), 1, 2)
    assert hit.tolist() == [False, True]
    assert v[1].tolist() == [5., 6., 7., 8.]
    assert not b.read(torch.tensor([1]), 0, 2)[1].any()
    assert not b.read(torch.tensor([1]), 3, 2)[1].any()
    assert b.allocated_bytes <= b.budget_bytes
    c = GradientBank(0.00004, (4,))
    c.load_state_dict(copy.deepcopy(b.state_dict()))
    assert torch.equal(c.read(torch.tensor([1]), 2, 2)[0], b.read(torch.tensor([1]), 2, 2)[0])


def test_observe_parity_and_feedback():
    torch.manual_seed(5)
    torch.set_num_threads(1)
    layer = torch.nn.Linear(8, 8)
    base = copy.deepcopy(layer)
    feedback = copy.deepcopy(layer)
    x, keys = torch.randn(5, 8), torch.arange(5)
    a = MechanisticMemory(layer, mib=.02, mode='observe', rank=4, topk=2)
    b = MechanisticMemory(feedback, mib=.02, mode='bpd', rank=4, topk=2)
    for step in range(2):
        for model in (base, layer, feedback):
            model.zero_grad()
        base(base(x)).square().mean().backward()
        with a.capture(keys, step, 0, 2):
            layer(layer(x)).square().mean().backward()
        with b.capture(keys, step, 0, 2):
            feedback(feedback(x)).square().mean().backward()
        assert torch.equal(layer.weight.grad, base.weight.grad)
        assert torch.equal(feedback.weight.grad, base.weight.grad) == (step == 0)
    assert b.admitted > 0
    assert b.bank.hits == 5
    assert not b.bank.pages[0][0].requires_grad


def test_resume_next_gradient():
    torch.manual_seed(19)
    x, keys = torch.randn(3, 8), torch.arange(3)
    a = torch.nn.Linear(8, 8)
    m = MechanisticMemory(a, mib=.02, rank=4)
    with m.capture(keys, 0, 0, 2):
        a(a(x)).square().mean().backward()
    b = copy.deepcopy(a)
    n = MechanisticMemory(b, mib=.02, rank=4)
    n.load_state_dict(copy.deepcopy(m.state_dict()))
    a.zero_grad()
    b.zero_grad()
    with m.capture(keys, 1, 0, 2):
        a(a(x)).square().mean().backward()
    with n.capture(keys, 1, 0, 2):
        b(b(x)).square().mean().backward()
    assert torch.equal(a.weight.grad, b.weight.grad)
