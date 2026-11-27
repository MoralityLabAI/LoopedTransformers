import torch

from research_gym.interp.molt_amortized_repair import (
    CONTROLS, LEARNED, Controller, bounded, charts, decision_loss, evaluate, fit,
    legal_logits, objective, prepare, root_pool, solve,
)
from research_gym.interp.molt_clock_repair import donor_world, margins
from research_gym.interp.molt_typed_clock import TypedClockDecoder


def test_disjoint_source_groups():
    train, dev = root_pool(4, 8, 4)
    def expand(roots):
        return {item for w in roots for item in (w, donor_world(w, "changed"), donor_world(w, "same"))}
    assert len(expand(train)) == 24
    assert len(expand(dev)) == 12
    assert not (expand(train) & expand(dev))


def test_controller_initialization_and_dose():
    torch.manual_seed(2)
    h = torch.randn(4, 64, dtype=torch.float64)
    q = torch.linalg.qr(torch.randn(64, 8, dtype=h.dtype)).Q
    d = h @ q @ q.T
    r = d.norm(dim=-1, keepdim=True)
    x = torch.randn(4, 145, dtype=h.dtype)
    net = Controller(x.shape[1], 8).double()
    torch.testing.assert_close(net(x, d, r, q), d, atol=1e-12, rtol=0)
    with torch.no_grad():
        net.net[-1].bias.fill_(100)
    assert bool((net(x, d, r, q).norm(dim=-1, keepdim=True) <= r + 1e-10).all())
    torch.testing.assert_close(bounded(d, torch.zeros_like(r)), torch.zeros_like(d))


def test_decision_solver_keeps_warm_start():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = TypedClockDecoder("addressed").double().eval()
    h = torch.randn(64, dtype=torch.float64)
    q = torch.linalg.qr(torch.randn(64, 8, dtype=h.dtype)).Q
    initial = torch.randn(8, dtype=h.dtype) * 0.1
    radius = initial.norm().reshape(1)
    target = margins(model, h + 2 * q @ initial).detach()
    winner = legal_logits(target).argmax()
    for kind in ("ce", "l2"):
        d, r = solve(model, h, target, q, radius, initial, kind)
        assert d.norm() <= radius + 1e-9
        assert r["final_objective"] <= r["initial_objective"]
        assert r["jacobian_calls"] == 8
        assert r["trial_readouts"] == 32
        value = objective(margins(model, h + d), target, winner, kind)
        assert abs(float(value.detach()) - r["final_objective"]) < 1e-10


def test_controller_training_and_full_replay():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = TypedClockDecoder("addressed").double().eval()
    original = {n: p.detach().clone() for n, p in model.state_dict().items()}
    v = 0.02 * torch.randn(256, 8, dtype=torch.float64)
    u = 0.02 * torch.randn(8, 64, dtype=torch.float64)
    q = torch.linalg.qr(u.T).Q
    a, b = root_pool(4, 8, 2)
    train = prepare(model, a, v, u)
    dev = prepare(model, b, v, u)
    bases = charts(model, train, q, 5)
    nets, mean, scale, receipts = fit(model, train, bases, updates=2, batch=4, seed=5)
    assert set(nets) == set(LEARNED)
    assert len({r["parameters"] for r in receipts}) == 1
    rows = evaluate(model, dev, bases, nets, mean, scale)
    assert len(rows) == len(dev) * (len(CONTROLS) + len(LEARNED))
    assert all(r["dose"] <= r["dose_cap"] + 1e-9 for r in rows)
    assert all(torch.equal(original[n], p) for n, p in model.state_dict().items())
