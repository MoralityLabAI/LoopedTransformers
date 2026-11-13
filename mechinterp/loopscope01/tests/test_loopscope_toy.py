import json

import numpy as np
import torch

from mechinterp.loopscope01.scripts import loopscope_toy as L

TINY = ["--n-nodes", "8", "--d", "32", "--layers", "1", "--heads", "2", "--steps", "6", "--batch", "16",
        "--max-train-depth", "3", "--ckpt-every", "3", "--log-every", "100", "--device", "cpu"]


def test_make_batch_hops_are_correct():
    tokens, hops = L.make_batch(8, 5, 3, 4)
    for b in range(5):
        edges = tokens[b, :16].view(8, 2)
        succ = {int(s): int(d) for s, d in edges}
        assert sorted(succ) == list(range(8))
        cur = int(tokens[b, 16])
        for j in range(4):
            cur = succ[cur]
            assert cur == int(hops[b, j])
    t2, h2 = L.make_batch(8, 5, 3, 4)
    assert torch.equal(tokens, t2) and torch.equal(hops, h2)


def test_ridge_probe_recovers_linear_labels():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(600, 16, generator=g)
    W = torch.randn(16, 4, generator=g)
    y = (X @ W).argmax(1)
    pred = L.ridge_probe(X[:500], y[:500], X[500:], 4)
    assert (pred == y[500:]).float().mean() > 0.7


def test_train_resume_eval_report(tmp_path):
    out = str(tmp_path)
    for lam in (0.0, 0.1):
        for seed in (1, 2):
            assert L.main(["train", "--lam", str(lam), "--seed", str(seed), "--out", out] + TINY) == 0
            assert L.main(["eval", "--lam", str(lam), "--seed", str(seed), "--out", out, "--device", "cpu",
                           "--eval-items", "32", "--max-eval-depth", "5", "--probe-train-items", "128"]) == 0
    rec = json.loads((tmp_path / "runs" / "lam0.1_s1" / "receipt.json").read_text())
    assert rec["steps"] == 6 and len(rec["checkpoint_sha256"]) == 64 and rec["params"] > 0
    # same params across arms
    rec0 = json.loads((tmp_path / "runs" / "lam0_s1" / "receipt.json").read_text())
    assert rec0["params"] == rec["params"]
    m = json.loads((tmp_path / "runs" / "lam0_s1" / "metrics.json").read_text())
    assert len(m["bits"]["cap"]["5"]) == 32 and set(m["summary"]) >= {"cap_heldout", "probe_all"}
    # 6-step models fail the amendment-2 validity gate, so the default report refuses a decision
    assert L.main(["report", "--out", out, "--n-boot", "50"]) == 2
    assert L.main(["report", "--out", out, "--n-boot", "50", "--valid-acc", "0"]) == 0
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["decision"] in ("trade-off (lam [0.1])", "no trade-off", "mixed")
    assert (tmp_path / "report.md").exists()


def test_resume_from_checkpoint_matches_straight_run(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    L.main(["train", "--lam", "0.1", "--seed", "1", "--out", a] + TINY)
    # first half: stop after the step-3 checkpoint by saving it through a 6-step run that is cut short
    orig = L.torch.save
    saved = {}

    def spy(obj, path):
        orig(obj, path)
        if str(path).endswith(".ckpt.pt"):
            raise KeyboardInterrupt

    L.torch.save = spy
    try:
        try:
            L.main(["train", "--lam", "0.1", "--seed", "1", "--out", b] + TINY)
        except KeyboardInterrupt:
            pass
    finally:
        L.torch.save = orig
    assert (tmp_path / "b" / "ckpt" / "lam0.1_s1.ckpt.pt").exists()
    L.main(["train", "--lam", "0.1", "--seed", "1", "--out", b] + TINY)
    sa = torch.load(tmp_path / "a" / "ckpt" / "lam0.1_s1.final.pt")["model"]
    sb = torch.load(tmp_path / "b" / "ckpt" / "lam0.1_s1.final.pt")["model"]
    for k in sa:
        assert torch.allclose(sa[k], sb[k], atol=1e-6), k
