"""NEC-03K exhaustive label-blind direction-matched SAE control tests."""
from __future__ import annotations

import math
from itertools import combinations
import pytest
import torch

from mechinterp.neuron_communities.nec03k_source_geometry import (
    ANGLE_LIMIT,SOURCE_CONTEXTS,exhaustive_direction_match,
)
from mechinterp.neuron_communities.nec03k_source_audit import (
    cosine,vector_sum,
)
from mechinterp.neuron_communities.topk_sae import TopKSAE
from mechinterp.neuron_communities.nec03g_worlds import worlds


def test_source_only_active_matching_is_exhaustive_directionally_nearest_and_disjoint():
    torch.set_num_threads(1)
    torch.manual_seed(103)
    sae=TopKSAE().eval()
    state=torch.randn(16,64)
    grad=torch.randn(16,32)
    grad=grad/grad.norm(dim=1,keepdim=True)
    down=torch.randn(32,64)
    rank=list(range(128))
    src,control,receipts=exhaustive_direction_match(
        sae,torch.zeros(64),torch.ones(64),state,rank,grad,down,count=2)
    assert src.shape==control.shape==(16,64)
    for i,row in enumerate(receipts):
        assert len(row["source_top_active"])==2
        assert len(row["selected_disjoint_control"])==2
        assert len(row["all_active"])==16
        assert set(row["source_top_active"]).isdisjoint(
            set(row["selected_disjoint_control"]))
        assert row["candidate_count"]==math.comb(14,2)
        pieces={p["id"]:p["vector"] for p in row["active_output_contributions32"]}
        source_cos=cosine(vector_sum(row["source_top_active"],pieces),
                          row["hub_gradient32"])
        gaps=[abs(cosine(vector_sum(ids,pieces),row["hub_gradient32"])-source_cos)
              for ids in combinations(
                  sorted(set(row["all_active"])-set(row["source_top_active"])),2)]
        assert abs(row["absolute_cosine_gap"]-min(gaps))<2e-5
        assert row["within_angle_gate"]==(
            row["absolute_cosine_gap"]<=ANGLE_LIMIT
            and row["gradient_norm"]>1e-7
            and row["physical_source_norm"]>1e-6
            and row["physical_control_norm"]>1e-6)


def test_deterministic_given_source_model_and_label_blind_gradient():
    torch.manual_seed(409)
    sae=TopKSAE().eval()
    x=torch.randn(3,64)
    g=torch.randn(3,32)
    W=torch.randn(32,64)
    options=(sae,torch.zeros(64),torch.ones(64),x,list(range(128)),g,W)
    a,b,r=exhaustive_direction_match(*options,count=3)
    c,d,s=exhaustive_direction_match(*options,count=3)
    assert torch.equal(a,c)
    assert torch.equal(b,d)
    assert r==s
    assert all(row["candidate_count"]==math.comb(13,3) for row in r)


def test_fixed_cardinality_allows_8_of_16_active_codes():
    torch.manual_seed(608)
    sae=TopKSAE().eval()
    states=torch.randn(2,64)
    grad=torch.randn(2,32)
    W=torch.randn(32,64)
    _,_,rows=exhaustive_direction_match(
        sae,torch.zeros(64),torch.ones(64),states,list(range(128)),grad,W,count=8)
    assert all(row["candidate_count"]==1 for row in rows)
    assert all(len(row["selected_disjoint_control"])==8 for row in rows)


def test_wrong_physical_dimension_is_rejected():
    torch.manual_seed(708)
    sae=TopKSAE().eval()
    x=torch.randn(2,64)
    with pytest.raises(ValueError):
        exhaustive_direction_match(sae,torch.zeros(64),torch.ones(64),
                                   x,list(range(128)),torch.randn(2,64),
                                   torch.randn(32,64),count=2)
    with pytest.raises(ValueError):
        exhaustive_direction_match(sae,torch.zeros(64),torch.ones(64),
                                   x,list(range(128)),torch.randn(2,32),
                                   torch.randn(32,64),count=9)


def test_original_confirmation_and_reserved_cohort_excluded_by_source_contract():
    assert SOURCE_CONTEXTS==16
    assert {worlds()[i]["cohort"] for i in range(8,16)}=={"development"}
    assert {worlds()[i]["cohort"] for i in range(64,128)}=={"confirmation"}
