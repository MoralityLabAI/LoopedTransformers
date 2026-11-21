"""NEC-03D: source-locked SAE and physical-dose unit tests on CPU."""
import hashlib

import pytest
import torch

from research.pef07.pef07 import Looper, as_tensor, source_split, model_digest
from mechinterp.neuron_communities.source_sae import (
    FEATURES, WIDTH, SourceSAE, clean_activations, physical_feature_delta,
    fit_source_sae, feature_rosters, save_sae, seal_array,
)
from mechinterp.neuron_communities.matched_dose import (
    donor_roster, graft_forward, match_dose, physical_dose, by_visit,
)
from mechinterp.neuron_communities.nec03_development import hub_disjoint_sham


def _model(seed=103):
    torch.manual_seed(seed)
    model=Looper().eval()
    return model


def test_native_sae_observes_12_visits_in_shared_64d_gelu_basis():
    torch.set_num_threads(1)
    model=_model()
    pool,_=source_split()
    original=model_digest(model)
    states=clean_activations(model,as_tensor(pool["train"][:32]))
    assert states.shape==(32,12,64)
    assert torch.isfinite(states).all()
    assert model_digest(model)==original


def test_sae_source_fit_has_no_weight_mutation_and_zero_reconstruction_graft():
    torch.set_num_threads(1)
    model=_model()
    pool,_=source_split()
    original=model_digest(model)
    sae,receipt,mean,scale=fit_source_sae(
        model,pool["train"][:16*80],seed=103,steps=2)
    assert receipt["n_contexts_train"] == 80
    assert receipt["n_holdout_contexts_seen_by_fit"] == 0
    assert receipt["source_activation_count"] == 80*16*12
    assert receipt["model_input_dimension"]==WIDTH
    assert 0 <= receipt["mean_active_features"] <= 128
    assert mean.shape==scale.shape==(64,)
    assert model_digest(model)==original
    a=torch.zeros(5,64)
    assert torch.equal(physical_feature_delta(sae,mean,scale,a,[]),a)
    delta=physical_feature_delta(sae,mean,scale,a,[0,2,4])
    assert delta.shape==(5,64) and torch.isfinite(delta).all()


def test_feature_selection_cardinality_and_sham_does_not_overlap():
    scores=torch.zeros((12,FEATURES),dtype=torch.float64)
    scores[0,4]=3
    scores[0,5]=2
    scores[9,7]=1
    selected,control=feature_rosters(scores,["0:2","0:7","9:11"])
    assert selected[0]==[4,5]
    assert selected[9]==[7]
    assert len(control[0])==2 and len(control[9])==1
    assert not set(selected[0])&set(control[0])
    with pytest.raises(ValueError):
        feature_rosters(scores,["12:0"])


def test_physical_dose_is_exact_with_no_clipping():
    W=torch.zeros((32,64))
    W[0,0]=2.
    W[1,1]=1.
    target=torch.zeros((3,64))
    control=torch.zeros_like(target)
    target[:,0]=torch.tensor([1.,2.,3.])
    control[:,1]=torch.tensor([2.,4.,6.])
    scaled,info=match_dose(target,control,W)
    assert torch.allclose(physical_dose(target,W),
                          physical_dose(scaled,W),atol=1e-6)
    assert all(info["matched_under_5pct"])
    assert all(not x for x in info["saturated"])
    assert info["max_gain"]==8.
    with pytest.raises(ValueError):
        match_dose(target,control,W,max_gain=-1)


def test_zero_dose_and_saturated_control_are_reported():
    W=torch.zeros(32,64)
    W[0,0]=1.
    target=torch.zeros((2,64))
    control=torch.zeros((2,64))
    target[0,0]=1.
    control[0,0]=0.001
    adjusted,receipt=match_dose(target,control,W)
    assert receipt["saturated"][0]
    assert not receipt["matched_under_5pct"][0]
    assert receipt["matched_under_5pct"][1]
    assert adjusted[1].abs().sum()==0


def test_only_memory_token_receives_visit_graft():
    torch.set_num_threads(1)
    model=_model()
    sources,_=source_split()
    bits=as_tensor(sources["train"][:16])
    activations=clean_activations(model,bits)
    sae=SourceSAE().eval()
    mean=torch.zeros(64)
    scale=torch.ones(64)
    donors,doses,hubs=donor_roster(
        model,activations,["0:1","5:2"],["0:3"],["0:4","5:7"],
        sae,mean,scale,
        {0:[0],5:[3]},{0:[10],5:[14]})
    assert set(donors)=={0,5}
    assert set(doses[0])=={"native_direct_dose","sham","sae","sae_sham"}
    assert len(doses[0]["native_direct_dose"])==16
    for mode in ("hubs","native","sham","sae","sae_sham"):
        logits=graft_forward(model,bits,donors,hubs,mode)
        assert logits.shape==(16,2)
        assert torch.isfinite(logits).all()


def test_rematched_native_sham_avoids_old_community_hubs():
    candidate={"nodes":["0:1","1:3"],"frequency_matched_sham":["0:7","1:5"]}
    graph={"hubs":["0:7","1:8"],
           "frequency":{"0:1":20,"1:3":18,"0:7":22,"1:5":17,
                        "0:4":21,"1:9":19}}
    sham=hub_disjoint_sham(candidate,graph)
    assert len(sham)==2
    assert not set(sham)&set(graph["hubs"])
    assert not set(sham)&set(candidate["nodes"])
    assert {x.split(":")[0] for x in sham}=={"0","1"}


def test_sealed_sae_receipt_checks_artifact_byte_digest(tmp_path):
    sae=SourceSAE()
    mean=torch.zeros(64)
    scale=torch.ones(64)
    sources={"fit":["00"],"dev":["11"],"heldout":[]}
    receipt={"fitted_trunk_model_sha256":"1"*64}
    sealed=save_sae(tmp_path/"sae.pt",sae,receipt,mean,scale,sources=sources)
    assert sealed["sae_checkpoint_sha256"]==hashlib.sha256(
        (tmp_path/"sae.pt").read_bytes()).hexdigest()
    assert len(sealed["source_roster_sha256"])==64
