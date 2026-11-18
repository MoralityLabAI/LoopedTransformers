"""B4-FACTOR-03 CPU regression. No heldout board contents or pretrained assets."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from mechinterp.bpd.adapters.trm import TRMBPDAdapter, build_mini_trm, default_trm_c_per_module
from mechinterp.bpd.ci import TiedCINetwork
from mechinterp.bpd.components import BPDModel
from mechinterp.bpd.config import CIConfig
from mechinterp.bpd.scripts import audit_b4_factor03_rebase as A


def test_fresh48_is_hash_locked_and_disjoint() -> None:
    ids = A.validate_fresh48(183)
    assert len(ids) == len(set(ids)) == 48
    assert ids[:4] == (71, 93, 37, 91)
    assert set(ids).isdisjoint(A.RESERVED64)
    assert set(ids).isdisjoint(A.validation_indices())
    with pytest.raises(AssertionError, match="source changed"):
        A.validate_fresh48(184)


def test_rebase_preserves_an_arbitrary_nonorthogonal_linear_map() -> None:
    torch.manual_seed(23)
    class Factors:
        C = 7
        V = torch.randn(11, 7)
        U = torch.randn(7, 9)
    w = Factors()
    # A strong shear keeps the dense operator but dramatically changes component geometry.
    q = torch.eye(7)
    q[0, 1] = 4.0
    w.V = w.V @ q
    w.U = torch.linalg.solve(q, w.U)
    vnew, unew, err = A.rebase_site(w)  # type: ignore[arg-type]
    assert err < 1e-6
    assert torch.allclose(vnew.double() @ unew.double(),
                          w.V.double() @ w.U.double(), rtol=1e-5, atol=1e-5)
    normalized = torch.nn.functional.normalize(unew, dim=-1)
    gram = normalized @ normalized.T
    offdiag = gram - torch.diag_embed(gram.diagonal())
    assert offdiag.abs().max() < 1e-4


def _mini():
    torch.manual_seed(53)
    base = build_mini_trm(
        hidden_size=32, num_heads=4, l_layers=2, h_cycles=1, l_cycles=2,
        puzzle_emb_len=4, seq_len=16, seed=113,
    ).float().eval()
    cm = default_trm_c_per_module(base, c=32)
    svd = BPDModel(copy.deepcopy(base), cm, max_passes=6, ablation=A.ZERO, init_svd=True)
    tr = BPDModel(copy.deepcopy(base), cm, max_passes=6, ablation=A.ZERO, init_svd=True)
    rb = BPDModel(copy.deepcopy(base), cm, max_passes=6, ablation=A.ZERO, init_svd=False)
    with torch.no_grad():
        for w in tr.wrappers.values():
            # The represented operator differs from the target, with residual retained
            # for the trained full-on identity control.
            w.V.add_(torch.randn_like(w.V) * 0.02)
            w.U.add_(torch.randn_like(w.U) * 0.02)
    errs = A.rebase_model(tr, rb)
    dims = {m: tr.wrappers[m].d_in for m in tr.module_order}
    ci = TiedCINetwork(
        dims, tr.c_per_module,
        CIConfig(variant="state_only", d_model=32, n_blocks=1, n_heads=4,
                 mlp_hidden=64, max_passes=16, max_seq_len=32),
    ).eval()
    return (
        svd, TRMBPDAdapter(svd.target_model, readout_at="outer_step"),
        tr, TRMBPDAdapter(tr.target_model, readout_at="outer_step"),
        rb, TRMBPDAdapter(rb.target_model, readout_at="outer_step"),
        ci, errs,
    )


def test_mini_recurrent_scoring_floors_parity_and_geometry() -> None:
    svd, ads, tr, adt, rb, adr, ci, errors = _mini()
    assert all(e < 1e-5 for e in errors.values())
    train = torch.randint(1, 11, (8, 16), dtype=torch.long)
    geo = A.training_geometry(tr, rb, adt, train, 19, 6)
    assert geo["invocations"] == [0]
    assert geo["n_training_boards"] == 8
    for m in tr.module_order:
        t, r = geo["sites"][m]["trained"], geo["sites"][m]["rebased"]
        assert t["reconstruction_residual_relative_fro"] > 1e-4
        assert r["reconstruction_residual_relative_fro"] == pytest.approx(
            t["reconstruction_residual_relative_fro"], abs=1e-5)
        assert r["mean_abs_offdiag_write_gram"] < 1e-4
        assert t["combined_error_rms_over_target"] >= 0

    x = torch.randint(1, 11, (3, 16), dtype=torch.long)
    rec = A.score_chunk(svd, ads, tr, adt, rb, adr, ci, x,
                        n_passes=6, c=32, k=19, readouts=(2, 5))
    assert set(rec["arms"]) == set(A.ARMS)
    for arm in A.ARMS:
        assert len(rec["arms"][arm]["pass_2"]["kl64"]) == 3
        assert len(rec["arms"][arm]["pass_5"]["argmax_flips"]) == 3
    for arm in ("F0", "F1", "F2"):
        assert max(abs(v) for v in rec["arms"][arm]["pass_5"]["kl64"]) < 1e-6
    assert max(abs(v) for v in rec["F4_vs_F3_kl64"]["pass_5"]) < 1e-5
    # The omitted residual changes the trained all-on operation.
    assert max(rec["arms"]["F3"]["pass_5"]["kl64"]) > 0


def test_paired_effect_and_decision_gate_are_deterministic() -> None:
    first = A._paired(np.array([0.01] * 48), np.array([2.0] * 48),
                      seed=A.BOOT_SEED, n_boot=256)
    assert first["n"] == 48
    assert first["ci95"][1] < 0
    assert first["effective_effect_n"] == pytest.approx(48)
    assert first == A._paired(np.array([0.01] * 48), np.array([2.0] * 48),
                              seed=A.BOOT_SEED, n_boot=256)
    arms = {}
    for name in A.ARMS:
        val = {
            "R1": 0.01, "T1": 2.0, "S2": 0.02, "S0": 0.5,
            "F0": 0.0, "F1": 0.0, "F2": 0.0, "F3": 1.0, "F4": 1.0,
            "T3": 0.5,
        }[name]
        arms[name] = {"pass_20": {"kl64": [val] * 48}}
    base = {"arms": arms, "F4_vs_F3_kl64": {"pass_20": [0.0] * 48},
            "operator_relative_error": {"site": 1e-7}}
    result = A.decide({"a": copy.deepcopy(base), "b": copy.deepcopy(base)},
                      readouts=(20,), n_boot=256)
    assert "improves" in result["primary_decision"]
    assert result["secondary_H2_noninferior"] is True
    broken = copy.deepcopy(base)
    broken["F4_vs_F3_kl64"]["pass_20"][0] = 0.1
    blocked = A.decide({"a": base, "b": broken}, readouts=(20,), n_boot=256)
    assert blocked["primary_decision"].startswith("BLOCKED")
    assert blocked["secondary_H2_noninferior"] is None
