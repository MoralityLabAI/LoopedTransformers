import copy

import torch
from torch.nn import functional as F

from research_gym.interp.aux_gym_v2_observable_bpd import run_write, trajectory
from research_gym.interp.aux_gym_v2_response_factor import (
    centered_weight, decode, gram_loss, normalized_trajectory, payload_report, teacher_pair,
)
from research_gym.interp.models import build_model


def test_gram_loss_with_trainable_different_heads():
    g = torch.Generator().manual_seed(13)
    a = torch.randn(2, 3, 8, dtype=torch.float64, generator=g, requires_grad=True)
    b = torch.randn(2, 3, 8, dtype=torch.float64, generator=g)
    ws = torch.randn(29, 8, dtype=torch.float64, generator=g, requires_grad=True)
    wt = torch.randn(29, 8, dtype=torch.float64, generator=g)
    cs = ws - ws.mean(0)
    ct = wt - wt.mean(0)
    dense = F.mse_loss(decode(a, cs), decode(b, ct))
    compact = gram_loss(a, b, cs, ct, ct.T @ ct)
    torch.testing.assert_close(dense, compact, atol=1e-12, rtol=0)
    left = torch.autograd.grad(dense, (a, ws), retain_graph=True)
    right = torch.autograd.grad(compact, (a, ws))
    for l, r in zip(left, right, strict=True):
        torch.testing.assert_close(l, r, atol=1e-12, rtol=0)
    assert left[1].norm() > 0


def test_actual_tied_decoder_loss_and_all_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    teacher = build_model("LOOP-T", hidden_size=16, num_heads=2, visits=4, vocab_size=32).double().eval()
    student = copy.deepcopy(teacher)
    with torch.no_grad():
        student.embedding.weight.add_(0.03 * torch.randn_like(student.embedding.weight))
    x = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    delta = 0.2 * torch.randn(2, 16, dtype=torch.float64)
    b, full = teacher_pair(teacher, x, 2, delta)
    wt = centered_weight(teacher).detach()
    torch.testing.assert_close(decode(b, wt), full, atol=1e-12, rtol=0)
    clean = student(x)
    patched, _ = run_write(student, x, 2, delta)
    a = normalized_trajectory(student, patched, 2) - normalized_trajectory(student, clean, 2)
    native = trajectory(student, patched, 2) - trajectory(student, clean, 2)
    ws = centered_weight(student)
    dense = F.mse_loss(native, full)
    compact = gram_loss(a, b, ws, wt)
    torch.testing.assert_close(native, decode(a, ws), atol=1e-12, rtol=0)
    torch.testing.assert_close(dense, compact, atol=1e-12, rtol=0)
    params = tuple(student.parameters())
    left = torch.autograd.grad(dense, params, retain_graph=True)
    right = torch.autograd.grad(compact, params)
    for l, r in zip(left, right, strict=True):
        torch.testing.assert_close(l, r, atol=1e-10, rtol=1e-8)
    assert dense > 0
    assert sum(float(g.norm()) for g in left) > 0


def test_payload_counts_head_and_gram():
    p = payload_report(32, 5, 128, 2048, 8)
    assert p["dense_response_bytes"] == 2621440
    assert p["factor_response_bytes"] == 163840
    assert p["response_payload_reduction_fraction"] == 0.9375
    assert p["frozen_centered_head_bytes"] == 2097152
    assert p["frozen_teacher_gram_bytes"] == 131072
    assert p["dense_plus_native_bytes"] == 2631680
    assert p["factor_plus_native_with_head_bytes"] == 2271232
    assert p["gram_plus_native_with_head_and_gram_bytes"] == 2402304
