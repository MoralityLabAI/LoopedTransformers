"""NEC-03F TopK16 source SAE and input-conditional route audit tests."""
import pytest
import torch

from research.pef07.pef07 import Looper,as_tensor,model_digest,source_split
from mechinterp.neuron_communities.topk_sae import (
    TopKSAE,fit_topk,active_feature_policy,
)
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.nec03f_active import build_pair
from mechinterp.neuron_communities.audit_nec03f import check_policy
from mechinterp.neuron_communities.matched_dose import graft_forward


def test_topk16_is_at_most_16_active_and_exactly_16_for_generic_codes():
    torch.manual_seed(103)
    sae=TopKSAE().eval()
    samples=torch.randn((20,64))
    with torch.no_grad():
        codes=sae.codes(samples)
        reconstruction,_=sae(samples)
    assert codes.shape==(20,128)
    assert reconstruction.shape==(20,64)
    assert ((codes>1e-7).sum(1)<=16).all()
    assert ((codes>1e-7).sum(1)==16).all()
    with pytest.raises(ValueError):
        TopKSAE(k=0)


def test_topk_train_only_uses_original_source_and_preserves_model():
    torch.set_num_threads(1)
    torch.manual_seed(131)
    model=Looper().eval()
    data,_=source_split()
    sha=model_digest(model)
    sae,receipt,mean,scale=fit_topk(model,data["train"][:80*16],seed=131,steps=3)
    assert receipt["train_contexts"]==80
    assert receipt["heldout_contexts_seen"]==0
    assert receipt["target_active_features_per_input"]==16
    assert receipt["observed_mean_active_features_per_input"]<=16
    assert mean.shape==scale.shape==(64,)
    assert model_digest(model)==sha


def test_label_blind_policy_picks_only_active_features_with_ranked_alternate():
    torch.manual_seed(167)
    sae=TopKSAE().eval()
    states=torch.randn((16,64))
    rank=list(range(128))
    source,control,rec=active_feature_policy(
        sae,torch.zeros(64),torch.ones(64),states,rank,n=1)
    assert source.shape==control.shape==(16,64)
    assert len(rec["top_source_ranked_active"])==16
    assert all(len(x)==1 for x in rec["bottom_source_ranked_active"])
    check_policy({"0":rec},[rank for _ in range(12)])
    forged=dict(rec)
    forged["top_source_ranked_active"]=[list(x) for x in rec["top_source_ranked_active"]]
    forged["top_source_ranked_active"][0]=rec["bottom_source_ranked_active"][0]
    with pytest.raises(AssertionError,match="selection differs"):
        check_policy({"0":forged},[rank for _ in range(12)])


def test_topk_matched_graft_is_source_visit_specific():
    torch.set_num_threads(1)
    torch.manual_seed(103)
    model=Looper().eval()
    src,_=source_split()
    bits=as_tensor(src["train"][:16])
    act=clean_activations(model,bits)
    sae=TopKSAE().eval()
    cand={"nodes":["0:1","5:3"]}
    policy={"hub_disjoint_native_sham":["0:7","5:9"]}
    rank=[list(range(128)) for _ in range(12)]
    for pair in (("native","sham"),("native","topk"),("topk","topk_alt")):
        donors,doses,hubs,choices=build_pair(
            model,act,cand,policy,rank,sae,
            torch.zeros(64),torch.ones(64),["0:6","5:6"],pair)
        assert set(donors)=={0,5}
        assert set(doses)==set(choices)=={0,5}
        for mode in ("hubs","native","sham"):
            logits=graft_forward(model,bits,donors,hubs,mode)
            assert logits.shape==(16,2)
            assert torch.isfinite(logits).all()


def test_policy_rejects_invalid_feature_budgets():
    sae=TopKSAE().eval()
    state=torch.ones((2,64))
    rank=list(range(128))
    with pytest.raises(ValueError):
        active_feature_policy(sae,torch.zeros(64),torch.ones(64),state,rank,n=0)
    with pytest.raises(ValueError):
        active_feature_policy(sae,torch.zeros(64),torch.ones(64),state,rank,n=9)
