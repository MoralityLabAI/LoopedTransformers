import torch

from experiments.molt_goblin_v1.observability_gym import (
    DelayChain,
    analytic,
    cut_states,
    observe,
    run,
    sample,
    train,
    world,
)
from lsa.observability import matched_sham, spectrum, stack_transport


def test_typed_stack_and_nested_kernels():
    # Instantaneous observation misses coordinate 2; the next visit sees it.
    eye = torch.eye(3, dtype=torch.float64)
    swap = eye[[1, 0, 2]]
    q = eye[:1]
    o = stack_transport((eye, swap), (q, q), eye)
    assert spectrum(o[:1])["d_obs"] == 1
    assert spectrum(o)["d_obs"] == 2
    assert spectrum(o)["null_dimension"] == 1
    torch.testing.assert_close(o, eye[:2])
    # Rectangular register interfaces and an injected support subspace.
    a = eye[:2]
    b = eye[:, :2]
    e = eye[:, 1:]
    out = stack_transport((a, b), (torch.eye(2, dtype=eye.dtype), q), e)
    assert out.shape == (3, 2)
    assert spectrum(out)["d_obs"] == 1


def test_distinct_sham_and_one_dimensional_nullspace():
    o = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    z = torch.tensor([0., 1., 0.], dtype=o.dtype)
    d = torch.tensor([1., 2., 3.], dtype=o.dtype)
    g = torch.Generator().manual_seed(5)
    s, receipt = matched_sham(d, z, o, g)
    assert receipt["status"] == "matched"
    torch.testing.assert_close(s, torch.tensor([1., 2., -3.], dtype=o.dtype))
    assert max(receipt[k] for k in ("signed_error", "norm_error", "radius_error")) < 1e-10
    _, fail = matched_sham(d, z, torch.eye(3, dtype=o.dtype), g)
    assert fail["status"] == "no_distinct_match"
    _, zero = matched_sham(torch.zeros_like(d), z, o, g)
    assert zero["status"] == "zero_dose"


def test_chain_jacobian_and_causal_cone():
    torch.manual_seed(7)
    model = DelayChain().double()
    x = sample(13, 1, 8, torch.float64)[0]
    _, bad = cut_states(model, x, 2)
    direct = torch.autograd.functional.jacobian(
        lambda d: observe(model, x, bad, 2, d)[0],
        torch.zeros(16, dtype=x.dtype),
    )
    exact = analytic(model, x, bad, 2)
    torch.testing.assert_close(exact, direct, atol=1e-10, rtol=0)
    cone = observe(model, x, bad, 2)[2]
    assert int(cone.sum()) == 5
    assert exact[~cone].abs().max() == 0
    assert spectrum(exact)["d_obs"] <= 5


def test_trained_gym_smoke_complete_roster():
    torch.set_num_threads(1)
    model = train(3, 2)
    basis = torch.eye(16, dtype=torch.float64)[:, :4]
    x = sample(19, 1, 8, torch.float64)[0]
    rows, audits = world(model, x, basis, 3, 0)
    assert len(rows) == 12
    assert audits["corrupt"]["analytic_autograd_max_abs"] < 1e-9
    for condition in audits.values():
        assert condition["terminal"]["d_obs"] <= condition["stack"]["d_obs"]
    report = run(2, 2, (3, 5))
    assert len(report["rows"]) == 96
    assert not report["promotion_authorized"]
    assert report["geometry"]["matched"] + report["geometry"]["zero_dose"] + report["geometry"]["unmatched"] == 16
