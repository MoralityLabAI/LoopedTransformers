import pytest
import torch

from research_gym.interp.molt_clock_carry import (
    chart, decode_carry, encode_carry, replay_case, run_cut, suffix,
)
from research_gym.interp.molt_typed_clock import MOD, TypedClockDecoder


def test_carry_chart_and_exact_suffix():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = TypedClockDecoder("addressed").double().eval()
    tokens = torch.tensor([[3, 1, 19, 2, 16]])
    with torch.no_grad():
        logits, carries, _, _ = run_cut(model, tokens, 2)
        p = carries[:, 1]
        c = encode_carry(p)
        assert c.shape == (1, MOD - 1)
        recovered = decode_carry(c)
        torch.testing.assert_close(p, recovered, atol=1e-14, rtol=0)
        q, s = suffix(model, recovered, tokens[:, 3:])
        torch.testing.assert_close(q, logits[:, 2:], atol=1e-10, rtol=0)
        torch.testing.assert_close(s, carries[:, 2:], atol=1e-10, rtol=0)
    e = chart(torch.float64)
    torch.testing.assert_close(e.T @ e, torch.eye(16, dtype=e.dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(e.sum(0), torch.zeros(16, dtype=e.dtype), atol=1e-12, rtol=0)
    with pytest.raises(ValueError):
        decode_carry(torch.ones(1, 16, dtype=e.dtype))


def test_fitted_factor_coordinate_replay_and_derivative():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = TypedClockDecoder("addressed").double().eval()
    tokens = torch.tensor([[3, 1, 19, 2, 16]])
    donor = tokens.clone()
    donor[0, 2] += 1
    v = 0.02 * torch.randn(256, 8, dtype=torch.float64)
    u = 0.02 * torch.randn(8, 64, dtype=torch.float64)
    q = torch.linalg.qr(u.T).Q
    r = replay_case(model, tokens, donor, v, u, q, 2)
    assert r["suffix_replay_max_abs"] < 1e-8
    assert r["local_carry_error"] < 1e-8
    assert r["suffix_carry_bytes"] == 128
    assert r["geometry"]["future"]["d_obs"] <= 16
    assert r["geometry"]["future_native_span"]["d_obs"] <= 8
    assert r["geometry"]["factorization_max_abs"] < 1e-8
