"""CPU unit tests on a tiny random Qwen3:  pytest tests -q   (needs checkpoints/tiny_base from make_tiny_model.py)"""
import os
import random
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from looprec import tasks
from looprec.adapters import LoRALinear
from looprec.laco import (build_student, collapse_blocks, count_trainable, laco_search_single, load_laco_manifest,
                          shallow_block)
from looprec.losses import distill_ce
from looprec.qwen3_core import Block, load_qwen3, rope_cos_sin


@pytest.fixture(scope="module")
def tiny():
    d = os.path.join(ROOT, "checkpoints", "tiny_base")
    if not os.path.exists(os.path.join(d, "model.safetensors")):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "make_tiny_model.py"), "--out", d])
    m, _ = load_qwen3(d, dtype=torch.float32)
    return m


MERGES = [{"anchor": 3, "absorbed": [4, 5, 6]}]


def test_tasks_are_correct():
    rng = random.Random(0)
    for _ in range(50):
        e = tasks.gen_fsm(rng, 12, "scratch")
        moves = e.prompt.split("\n")[0].split()[1:]
        s = 0
        for m_, t in zip(moves, e.target.split("\n")[0].split(",")):
            s = tasks._PERMS[m_][s]
            assert int(t) == s
        assert e.target.split("State:")[1].strip() == str(s)
        a = tasks.gen_add(rng, 5, "scratch")
        x, y = a.prompt[2:-3].split("+")
        assert a.target.split("|")[1].strip() == str(int(x) + int(y))
        c = tasks.gen_chain(rng, 6, "direct")
        assert c.target.strip().isdigit()
        m = tasks.gen_mul(rng, (3, 2), "scratch")
        p, q = m.prompt[2:-3].split("*")
        assert sum(int(t) for t in m.target.split("|")[0].split(",")) == int(p) * int(q)


def test_collapse_modes(tiny):
    blocks = list(tiny.entries)
    last = collapse_blocks(blocks, 3, [4, 5], {"weight": "last"})
    assert torch.equal(last.q_proj.weight, blocks[5].q_proj.weight)
    first = collapse_blocks(blocks, 3, [4, 5], {"weight": "first"})
    assert torch.equal(first.q_proj.weight, blocks[3].q_proj.weight)
    rd = collapse_blocks(blocks, 3, [4, 5], {"weight": "rdsc"})
    ref = blocks[3].q_proj.weight + (blocks[4].q_proj.weight - blocks[3].q_proj.weight) + (blocks[5].q_proj.weight - blocks[3].q_proj.weight)
    assert torch.allclose(rd.q_proj.weight, ref, atol=1e-6)
    assert torch.equal(rd.ln1.weight, blocks[3].ln1.weight)          # norm_mode=base keeps the anchor norm


def test_recurrent_site_invariants(tiny):
    rec = dict(max_loops=4, adapter="loop", rank=4, alpha=8, norm="loop", gate=True, init="laco")
    st = build_student(tiny, MERGES, rec=rec).eval()
    plain = build_student(tiny, MERGES).eval()
    ids = torch.randint(0, 1000, (2, 24))
    with torch.no_grad():
        o1, _ = st.forward_hidden(ids, n_loops=1)
        o0, _ = plain.forward_hidden(ids)
        assert torch.allclose(o1, o0, atol=1e-5), "fresh adapters/gates must make T=1 identical to the plain collapsed student"
        o2, states = st.forward_hidden(ids, n_loops=2, collect=True)
        assert not torch.allclose(o2, o1)
        steps = [s for s in states if s[2] in ("step", "site_final")]
        assert len(steps) == 2 and all(s[1].shape == (2, 24, tiny.cfg.hidden_size) for s in steps)
        # recurrence is explicit: output of step t+1 == step t + F(step t)
        site = st.site()
        cos, sin = rope_cos_sin(24, tiny.cfg.head_dim, tiny.cfg.rope_theta, ids.device, torch.float32)
        h1 = steps[0][1]
        site.ctx.t = 1
        assert torch.allclose(steps[1][1], h1 + site.gate[1] * site.block.delta(h1, cos, sin), atol=1e-5)
        # T beyond trained loops is supported by clamping the adapter index
        st.forward_hidden(ids, n_loops=9)


def test_teacher_untouched_by_student_lora(tiny):
    w0 = tiny.entries[0].q_proj.weight.clone()
    st = build_student(tiny, MERGES, lora=dict(rank=4, alpha=8))
    assert isinstance(st.entries[0].q_proj, LoRALinear)
    assert not isinstance(tiny.entries[0].q_proj, LoRALinear)
    assert torch.equal(tiny.entries[0].q_proj.weight, w0)
    assert count_trainable(st) > 0 and all(not p.requires_grad for p in tiny.parameters())


def test_param_matching_and_flops(tiny):
    rec = dict(max_loops=4, adapter="loop", rank=4, alpha=8, norm="loop", gate=True)
    st = build_student(tiny, MERGES, rec=rec)
    blk = st.site().block
    n_ad = 4 * 4 * sum(getattr(blk, k).in_features + getattr(blk, k).out_features for k in Block.LINEARS)
    assert count_trainable(st) == n_ad + 4 * (2 * 64 + 2 * 16) + 4
    f1, f4 = st.flops_per_token(256, 1), st.flops_per_token(256, 4)
    assert f4 > f1
    assert st.n_layer_evals(4) - st.n_layer_evals(1) == 3


def test_laco_single_search_picks_valid_range(tiny):
    calib = torch.randint(0, 1000, (4, 32))
    merges, table = laco_search_single(tiny, calib, 3, bs=4)
    m = merges[0]
    assert m["absorbed"] == list(range(m["anchor"] + 1, m["anchor"] + 4))
    assert m["anchor"] >= 2 and m["absorbed"][-1] <= tiny.cfg.num_layers - 2


def test_distill_loss_zero_when_equal(tiny):
    h = torch.randn(2, 10, tiny.cfg.hidden_size, requires_grad=True)
    ids = torch.randint(0, 1000, (2, 10))
    loss, st = distill_ce(h, h.detach(), ids, torch.ones(2, 10, dtype=torch.bool), tiny.head_weight(), 1.0, 0.0, chunk=7)
    assert abs(st["kl"]) < 1e-5
    loss.backward()
    assert h.grad is not None


def test_lab_recipe_ingest(tmp_path):
    import json
    rec = {"source_layers": 28, "ops": [], "weight_mode": "rdsc", "norm_mode": "base", "qknorm_mode": "follow",
           "layout": [[i] for i in range(10)] + [[10, 11, 12]] + [[i] for i in range(13, 16)] + [[i] for i in range(18, 28)]}
    p = tmp_path / "laco_recipe.json"
    p.write_text(json.dumps(rec))
    m = load_laco_manifest(str(p))
    assert m["merges"] == [{"anchor": 10, "absorbed": [11, 12]}]
    assert m["deleted"] == [16, 17] and m["modes"]["weight"] == "rdsc" and m["kept_layer_count"] == 24
