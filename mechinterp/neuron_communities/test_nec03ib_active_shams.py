"""NEC-03I-B label-blind hash active-feature sham tests."""
import json

import pytest
import torch

from mechinterp.neuron_communities.nec03ib_active_shams import (
    VARIANTS, scrambled_rank, source_and_hash_donors,
    pair_donors, load_frozen_source, check_logits,
)
from mechinterp.neuron_communities.nec03ib_audit import verify_route
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.topk_sae import TopKSAE
from mechinterp.neuron_communities.matched_dose import graft_forward
from mechinterp.neuron_communities.nec03g_worlds import worlds
from research.pef07.pef07 import Looper, source_split, as_tensor


def test_hash_rank_is_total_deterministic_source_blind_and_variant_distinct():
    perms=[scrambled_rank(13007,5,i) for i in VARIANTS]
    assert len(perms)==4
    assert all(len(p)==128 and sorted(p)==list(range(128)) for p in perms)
    assert len({tuple(x) for x in perms})==4
    assert scrambled_rank(13007,5,0)==perms[0]
    assert scrambled_rank(13007,6,0)!=perms[0]
    with pytest.raises(ValueError):
        scrambled_rank(1,12,0)
    with pytest.raises(ValueError):
        scrambled_rank(1,0,4)


def test_hash_control_is_active_and_disjoint_from_source_top_on_each_input():
    torch.manual_seed(103)
    sae=TopKSAE().eval()
    states=torch.randn(16,64)
    rank=list(range(128))
    a,b,record=source_and_hash_donors(
        sae,torch.zeros(64),torch.ones(64),states,rank,n=2,
        hash_rank=scrambled_rank(104,5,2))
    assert a.shape==b.shape==(16,64)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    for i in range(16):
        active=set(record["all_active_feature_indices"][i])
        top=set(record["source_top_active"][i])
        control=set(record["hash_selected_disjoint_active"][i])
        assert len(active)==16
        assert len(top)==len(control)==2
        assert top.isdisjoint(control)
        assert top.issubset(active) and control.issubset(active)
    verify_route({"5":record},[rank for _ in range(12)],104,2)
    altered=dict(record)
    altered["hash_selected_disjoint_active"]=[
        list(x) for x in record["hash_selected_disjoint_active"]]
    altered["hash_selected_disjoint_active"][0]=list(record["source_top_active"][0])
    with pytest.raises(AssertionError):
        verify_route({"5":altered},[rank for _ in range(12)],104,2)


def test_pair_local_dose_and_graph_nodes_share_original_visit_geometry():
    torch.set_num_threads(1)
    torch.manual_seed(131)
    model=Looper().eval()
    source,_=source_split()
    bits=as_tensor(source["train"][:16])
    acts=clean_activations(model,bits)
    sae=TopKSAE().eval()
    rank=[list(range(128)) for _ in range(12)]
    cand={"nodes":["2:4","5:8"],"id":"dev-synthetic"}
    for variant in VARIANTS:
        donors,gates,physical,source_choices=pair_donors(
            model,acts,cand,["2:9"],sae,
            torch.zeros(64),torch.ones(64),
            rank,101,variant)
        assert set(donors)==set(gates)==set(physical)==set(source_choices)=={2,5}
        for mode in ("hubs","native","sham"):
            values=graft_forward(model,bits,donors,gates,mode)
            assert values.shape==(16,2)
        for site in physical.values():
            assert all(x<=1.0+1e-6 for x in site["left_gain"])
            assert all(x<=1.0+1e-6 for x in site["right_gain"])


def test_original_logits_crosscheck_fails_closed():
    x=torch.tensor([[1.,-1.],[-1.,1.]])
    check_logits(x,[[1.,-1.],[-1.,1.]])
    with pytest.raises(AssertionError,match="original"):
        check_logits(x,[[2.,-1.],[-1.,1.]])


def test_confirmation_cohort_refusal_occurs_before_loading_checkpoint(tmp_path):
    p=tmp_path/"root"
    p.mkdir()
    (p/"world_spec.json").write_text(json.dumps(worlds()[64]))
    with pytest.raises(PermissionError,match="previously exposed"):
        load_frozen_source(p,tmp_path/"missing-lock")
