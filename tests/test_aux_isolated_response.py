import sys

import pytest
import torch
from torch.nn import functional as F

from research_gym.interp.aux_gym_v2 import trace_loss
from research_gym.interp.aux_gym_v2_isolated_response import phase
from research_gym.interp.aux_gym_v2_isolated_worker import answer_states, task_loss
from research_gym.interp.aux_gym_v2_observable_bpd import run_write
from research_gym.interp.models import build_model


def test_native_answer_only_path_and_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = build_model("LOOP-T", hidden_size=16, num_heads=2, visits=4, vocab_size=32).double()
    tokens = torch.tensor([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]])
    target = torch.tensor([2, 4])
    trace = torch.tensor([[1, 2, 3, 2], [4, 3, 1, 4]])
    delta = 0.2 * torch.randn(2, 16, dtype=torch.float64)
    native = model(tokens)
    patched, _ = run_write(model, tokens, 2, delta)
    a = answer_states(model, tokens)
    b = answer_states(model, tokens, 2, delta)
    read = F.linear(a, model.embedding.weight)
    read_patched = F.linear(b, model.embedding.weight)
    expected = torch.stack([model.lens(h[:, -1]) for h in native.visit_outputs], dim=1)
    expected_patched = torch.stack([model.lens(h[:, -1]) for h in patched.visit_outputs], dim=1)
    torch.testing.assert_close(read, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(read_patched, expected_patched, atol=1e-12, rtol=0)
    lhs = trace_loss(model, native, target, trace, "symbolic_trace", 0.5)
    rhs = task_loss(model, a, target, trace)
    # Compare a nonzero differentiable response penalty too.
    lhs = lhs + (expected_patched - expected).square().mean()
    rhs = rhs + (read_patched - read).square().mean()
    torch.testing.assert_close(lhs, rhs, atol=1e-8, rtol=0)
    params = tuple(model.parameters())
    l = torch.autograd.grad(lhs, params, retain_graph=True)
    r = torch.autograd.grad(rhs, params)
    for x, y in zip(l, r, strict=True):
        torch.testing.assert_close(x, y, atol=1e-8, rtol=1e-7)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervisor")
def test_worker_limits_and_timeout(tmp_path):
    code = (
        "import resource; "
        "assert resource.getrlimit(resource.RLIMIT_AS)==(1073741824,1073741824); "
        "assert resource.getrlimit(resource.RLIMIT_CPU)==(4,4); "
        "print('bounded')"
    )
    receipt = phase([sys.executable, "-c", code], tmp_path / "ok.log",
                    address_mib=1024, rss_mib=512, wall_seconds=4, cpu_seconds=4)
    assert receipt["status"] == "success"
    assert not receipt["hard_process_tree_rss_cap"]
    assert "bounded" in (tmp_path / "ok.log").read_text()
    timed = phase([sys.executable, "-c", "import time; time.sleep(3)"], tmp_path / "time.log",
                  address_mib=1024, rss_mib=512, wall_seconds=0.15, cpu_seconds=4)
    assert timed["status"] == "wall_timeout"
