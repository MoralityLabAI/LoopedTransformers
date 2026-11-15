"""Mathematical and model-level invariants for PEF-08 frozen interventions."""
import hashlib
import random
import torch
import pytest

from research.pef07 import pef07 as p7
from research.pef08.pef08 import (
    PAIRS, source_data, labels, states_by_loop, group_delta,
    pair_margin, choose_channels, inject_pair_erasure, replay,
    native_margin, ridge_fit, ridge_margin,
)


def test_source_contexts_complete_and_disjoint():
    catalog, signature, manifest = source_data()
    assert len(signature) == 64
    assert {name:len(rows) for name,rows in catalog.items()} == {
        "train":2560,"probe":768,"heldout":768}
    assert set(map(tuple,catalog["train"].tolist())).isdisjoint(
        set(map(tuple,catalog["heldout"].tolist())))
    for rows in catalog.values():
        for off in range(0,len(rows),16):
            group = rows[off:off+16]
            assert len({tuple(r[4:].tolist()) for r in group}) == 1
            assert [tuple(r[:4].tolist()) for r in group] == [
                tuple((m>>j)&1 for j in range(4)) for m in range(16)]


def test_erasure_exactly_removes_selected_hidden_mixed_term():
    rng = torch.Generator().manual_seed(909)
    x = torch.randn(3*16,32,generator=rng)
    for pair in PAIRS:
        selected = (1,3,5,9,12,16,20,29)
        copy = x.clone()
        out = inject_pair_erasure(x,pair,selected)
        assert torch.equal(x,copy)
        assert torch.equal(inject_pair_erasure(x,pair,()),x)
        before,after = group_delta(x,pair),group_delta(out,pair)
        assert torch.allclose(after[:,list(selected)],
                              torch.zeros_like(after[:,list(selected)]),atol=1e-6)
        remaining = sorted(set(range(32))-set(selected))
        assert torch.equal(after[:,remaining],before[:,remaining])
        a,b = pair
        for mask in range(16):
            if mask & ((1<<a)|(1<<b)) != ((1<<a)|(1<<b)):
                assert torch.equal(out.reshape(3,16,32)[:,mask],
                                   x.reshape(3,16,32)[:,mask])


def test_zero_channel_intervention_is_identical_natural_replay():
    torch.manual_seed(103)
    cls = p7.Looper if hasattr(p7,"Looper") else p7.CoalitionMemoryLooper
    model = cls().eval()
    catalog,_,_ = source_data()
    bits = catalog["heldout"][:32]
    with torch.no_grad():
        states = states_by_loop(model,bits)
        for t in (4,8,12):
            patched = inject_pair_erasure(states[t],(0,1),())
            assert torch.equal(replay(model,bits,patched,t,12),states[12])
            assert torch.equal(replay(model,bits,patched,t,14),states[14])


def test_frozen_ridge_margin_agrees_with_original_fitted_readout():
    torch.manual_seed(117)
    cls = p7.Looper if hasattr(p7,"Looper") else p7.CoalitionMemoryLooper
    model=cls().eval()
    cat,_,_=source_data()
    with torch.no_grad():
        probe=states_by_loop(model,cat["probe"][:64])[14]
        held=states_by_loop(model,cat["heldout"][:64])[14]
        fit=ridge_fit(model,probe,labels(cat["probe"][:64],"coalition"))
        assert torch.equal((ridge_margin(held,fit)>=0).long(),p7.apply_ridge(held,fit))


def test_channels_selected_deterministically_from_probe_only():
    state=torch.arange(16*32,dtype=torch.float32).reshape(16,32)
    w=torch.ones(32)
    a=choose_channels(state,(0,1),w,103,"coalition",12)
    b=choose_channels(state.clone(),(0,1),w,103,"coalition",12)
    assert a==b
    assert a[0]==tuple(range(8))  # all tied scores, lower index wins
    assert len(a[2])==8
    assert all(len(set(mask))==8 for mask in a[2])
    assert choose_channels(state,(0,1),w,103,"threshold",12)[2]!=a[2]
