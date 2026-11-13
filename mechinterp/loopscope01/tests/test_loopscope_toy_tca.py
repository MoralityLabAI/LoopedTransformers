import numpy as np
import torch

from mechinterp.loopscope01.scripts import loopscope_toy as L
from mechinterp.loopscope01.scripts import loopscope_toy_tca as T


def tiny():
    torch.manual_seed(0)
    m = L.LoopedChase(n_nodes=8, d=32, layers=1, heads=2).double()
    m.eval()
    return m


def test_delta_matches_finite_difference():
    torch.set_default_dtype(torch.float64)
    try:
        m = tiny()
        comps = {k: tuple(x.double() for x in v) for k, v in T.svd_components(m).items()}
        res = T.tca(m, comps, depth=3, items=8, seed=5)
        mats = T.matrices(m)
        clean = torch.as_tensor(res["clean_margin"])
        for mi in (0, 3, 5):
            name, mod, sl = mats[mi]
            U, S, V = comps[name]
            for t in (1, 3):
                eps = 1e-4
                comps_eps = dict(comps)
                comps_eps[name] = (U, S * eps, V)
                meas = (T.ablate(m, mats, comps_eps, res["tokens"], res["target"], 3, mi, 0, {t}) - clean).numpy() / eps
                pred = res["delta"][mi, t - 1, :, 0]
                np.testing.assert_allclose(meas, pred, rtol=2e-3, atol=1e-6)
    finally:
        torch.set_default_dtype(torch.float32)


def test_margin_is_log_odds():
    logits = torch.tensor([[2.0, 0.0, 0.0]])
    y = T.margin(logits, torch.tensor([0]))
    p = torch.softmax(logits, 1)[0, 0]
    assert torch.isclose(y[0], torch.log(p / (1 - p)))


def test_null_sd_matches_random_rotations():
    g = torch.Generator().manual_seed(1)
    X, A = torch.randn(10, 6, generator=g), torch.randn(10, 4, generator=g)
    u = torch.randn(4, generator=g)
    u /= u.norm()
    sd = (X.T @ (A @ u)).norm() / np.sqrt(6)
    vs = torch.randn(20000, 6, generator=g)
    vs /= vs.norm(dim=1, keepdim=True)
    d = -(vs @ X.T @ A @ u)
    assert abs(float(d.std()) / float(sd) - 1) < 0.03


def test_controls_shapes_and_rotation_equals_raw_when_v_given():
    m = tiny()
    comps = {k: tuple(x.double() for x in v) for k, v in T.svd_components(m).items()}
    mats = T.matrices(m)
    tokens, hops = L.make_batch(8, 6, 2, 3)
    target = hops[:, 2]
    V = comps[mats[0][0]][2]
    a = T.ablate(m, mats, comps, tokens, target, 3, 0, 1, {2})
    b = T.ablate(m, mats, comps, tokens, target, 3, 0, 1, {2}, V[:, 1])
    assert torch.allclose(a, b)
    res = T.controls(m, comps, 3, 6, 2, n_top=2, n_rot=2)
    assert len(res["by_loop"]) == 3 and len(res["cells"]) == len(mats) * 2 * 3
