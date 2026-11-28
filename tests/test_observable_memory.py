from dataclasses import replace

import pytest
import torch

from lsa.observable_memory import MapBinding, MapCache, Sketch, tensor_digest
from experiments.molt_goblin_v1.mms_observer import run


def make_binding(epoch="e1"):
    return MapBinding("a" * 64, "root1", epoch, "b" * 64, "c" * 64,
                      "d" * 64, 2, 3, "e" * 64, 2)


def test_sketch_roundtrip_and_tail():
    o = torch.diag(torch.tensor([4., 2., 1., 0.], dtype=torch.float64))
    sketch = Sketch.build(o, 2)
    restored = Sketch.decode(sketch.encode())
    d = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    for k in (0, 1, 2):
        p = restored.project(d, k)
        torch.testing.assert_close(p, sketch.project(d, k))
        assert (o @ (d - p)).norm() <= restored.tail(k) * (d - p).norm() + 1e-10
    with pytest.raises(ValueError):
        Sketch.decode(sketch.encode()[:-1])
    assert tensor_digest(d) != tensor_digest(d + 1)


def test_binding_and_epoch_invalidation():
    b = make_binding()
    sketch = Sketch.build(torch.eye(4, dtype=torch.float64), 2)
    size = len(b.key()) + len(sketch.encode())
    cache = MapCache(size)
    cache.reset_scope(b.checkpoint, b.epoch)
    cache.get(b, lambda: sketch)
    cache.get(b, lambda: pytest.fail("cache hit must not rebuild"))
    for field, value in (("frame", "root2"), ("observer", "f" * 64),
                         ("state_digest", "f" * 64), ("input_digest", "f" * 64),
                         ("position", 4), ("cut", 3), ("support", "f" * 64),
                         ("max_rank", 1), ("epoch", "e2"), ("checkpoint", "f" * 64)):
        assert replace(b, **{field: value}).key() != b.key()
    b2 = replace(b, position=4)
    cache.get(b2, lambda: sketch)
    assert cache.receipt()["evictions"] == 1
    assert cache.receipt()["retained_bytes"] == size
    cache.reset_scope(b.checkpoint, "e2")
    assert cache.receipt()["invalidations"] == 1
    with pytest.raises(ValueError):
        cache.get(b, lambda: sketch)
    cache.get(make_binding("e2"), lambda: sketch)
    assert cache.receipt()["hits"] == 1
    assert cache.receipt()["builds"] == 3


def test_zero_budget_and_oversize():
    b = make_binding()
    s = Sketch.build(torch.eye(4, dtype=torch.float64), 2)
    for cap in (0, 1):
        c = MapCache(cap)
        c.reset_scope(b.checkpoint, b.epoch)
        c.get(b, lambda: s)
        c.get(b, lambda: s)
        assert c.receipt()["builds"] == 2
        assert c.receipt()["bypasses"] == 2
        assert c.receipt()["peak_retained_bytes"] == 0


def test_full_driver_replay_and_budget():
    report = run(2, 2, (7,))
    assert report["world_roots"] == 4
    assert len(report["rank_rows"]) == 4 * 3 * 16
    baseline = report["cache_summary"][0]
    for cell in report["cache_summary"]:
        assert cell["verified"] == baseline["verified"]
        assert cell["queries"] == 16
        assert cell["peak_retained_bytes"] <= cell["cap_bytes"]
        assert cell["max_replay_error"] <= 1e-10
        assert cell["builds"] + cell["hits"] == 16
    assert report["cache_summary"][-1]["hits"] == 8
    assert report["cache_summary"][-1]["invalidations"] == 4
    assert not report["promotion_authorized"]
