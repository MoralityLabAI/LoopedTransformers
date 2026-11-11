import json
import math
import os
from pathlib import Path

import pytest
import torch

torch.set_num_threads(2)
ls = pytest.importorskip("research_gym.scripts.loopscope_lean")
from lcgrt.model import ModelConfig, ProofNet  # noqa: E402
from lcgrt.encode import N_FEATS, N_EDGE_TYPES  # noqa: E402


def tiny(slow, T=4, seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(arch="hrm", width=32, heads=4, fast_steps=T, slow_every=2, slots=3, slow=slow)
    return ProofNet(cfg).double().eval()


def batch(B=3, N=7, seed=1):
    g = torch.Generator().manual_seed(seed)
    et = torch.randint(0, N_EDGE_TYPES, (B, N, N), generator=g)
    et[:, torch.arange(N), torch.arange(N)] = 1
    legal = torch.rand(B, N, generator=g) > 0.4
    legal[:, 0] = True
    return dict(x=torch.randn(B, N, N_FEATS, generator=g).double(), ttype=torch.randint(0, 5, (B, N), generator=g),
                etype=et, tok_mask=torch.ones(B, N, dtype=torch.bool), legal=legal,
                optimal=legal.clone(), main=torch.zeros(B, dtype=torch.long))


class Ex:
    """Minimal stand-in so analyze() can collate: reuse a prebuilt batch."""


@pytest.mark.parametrize("slow", [False, True])
def test_forward_states_matches_model(slow):
    m = tiny(slow, T=6)
    b = batch()
    with torch.no_grad():
        mine = ls.forward_states(m, b, 6)
        ref = m(b, fast_steps=6, glob=torch.zeros(3, 5).double())
    k = 2
    ticks = [mine[t] for t in range(6) if (t + 1) % k == 0 or t == 5]
    assert len(ticks) == len(ref)
    for a, r in zip(ticks, ref):
        assert torch.allclose(a, r["logits"], atol=1e-9)


@pytest.mark.parametrize("slow", [False, True])
def test_tca_matches_finite_difference_and_gram(slow):
    m = tiny(slow, T=4)
    for p in m.parameters():
        p.requires_grad_(False)
    if slow:
        m.slot_init.requires_grad_(True)
    b = batch()
    b["x"].requires_grad_(True)
    m.fast.att.o.weight.requires_grad_(True)
    tca = ls.TCA(m, K=4)
    tca.reset()
    logits = ls.forward_states(m, b, 4)
    final = logits[-1].argmax(-1)
    lp = torch.log_softmax(logits[-1], -1).gather(1, final[:, None]).squeeze(1)
    tca.rows.clear()
    lp.sum().backward(retain_graph=True)
    tca.remove()
    # every tracked matrix has one row per call; slow clock has T/2 calls
    for name, (mod, clock) in tca.mods.items():
        calls = sorted(c for (n, c) in tca.rows if n == name)
        # the last slow tick's output never reaches the readout, so it carries no gradient and is not recorded
        want = 4 if clock == "fast" else 1
        assert calls == list(range(want)), (name, calls)
    # first-order check: summed over calls, -s * Delta_unit equals the directional derivative of log p
    # along "remove component k of fast.att.o", i.e. -s u^T (dlogp/dW) v from plain autograd
    name = "fast.att.o"
    mod = tca.mods[name][0]
    U, S, V = tca.bases[name]
    k = 1
    tot = sum(tca.rows[(name, c)][0]["delta"][:, k] for c in range(4)) * S[k].item()
    per_item = []
    for i in range(lp.shape[0]):
        mod.weight.grad = None
        lp[i].backward(retain_graph=True)
        per_item.append(-S[k] * (U[:, k] @ mod.weight.grad @ V[:, k]))
    assert torch.allclose(torch.stack(per_item), tot, rtol=1e-4, atol=1e-7), (per_item, tot)
    # F equals the explicit norm of the summed per-call Gram form: check z definitions are consistent
    for (n, c), rs in tca.rows.items():
        r = rs[0]
        assert torch.all(r["F"] >= 0) and torch.isfinite(r["zr"]).all()
        d_in = tca.mods[n][0].in_features
        d_out = tca.mods[n][0].out_features
        # z_both = delta / (F / sqrt(d_in d_out))
        assert torch.allclose(r["zb"], r["delta"] / (r["F"][:, None] / math.sqrt(d_in * d_out) + ls.EPS), rtol=1e-6)


def test_gram_fragility_equals_explicit():
    X = torch.randn(2, 5, 6, dtype=torch.double)
    A = torch.randn(2, 5, 4, dtype=torch.double)
    gram = ((X @ X.transpose(1, 2)) * (A @ A.transpose(1, 2))).sum((1, 2)).sqrt()
    explicit = torch.stack([(X[i].T @ A[i]).norm() for i in range(2)])
    assert torch.allclose(gram, explicit)


def test_report_roundtrip(tmp_path):
    m = tiny(True, T=4)
    exs = None
    # drive analyze through a fake collate-free path: monkeypatch collate to return the prebuilt batch
    import lcgrt.encode as enc
    b = batch(B=4)
    old = enc.collate
    enc.collate = lambda e, dev: dict(b, x=b["x"].clone())
    try:
        class E:
            gold_tok = 0
        res = ls.analyze(m, [E() for _ in range(4)], 4, batch=4, K=4)
    finally:
        enc.collate = old
    res.update(ckpt="tiny", model=m.cfg.to_json())
    (tmp_path / "tiny_T4.json").write_text(json.dumps(res))
    ls.report(type("A", (), {"dir": str(tmp_path)})())
    assert (tmp_path / "report.md").exists() and (tmp_path / "decodability_fragility.png").exists()
    assert len(res["decod"]) == 4 and {c["clock"] for c in res["tca"]} == {"fast", "slow"}


@pytest.mark.skipif(not (ls.RUN_C1 / "Sloop_s17b" / "latest.pt").exists(), reason="checkpoint not on this machine")
def test_real_checkpoint_smoke(tmp_path):
    m = ls.load_model("Sloop_s17b", "cpu")
    exs = ls.load_examples(n_theorems=4, n_states=4, seed=0)
    assert len(exs) == 4
    res = ls.analyze(m, exs, 4, batch=4, K=8)
    assert res["decod"][-1]["agree_final"] == 1.0
    assert all(c["F_mean"] > 0 for c in res["tca"])
