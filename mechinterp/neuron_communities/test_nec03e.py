"""NEC-03E attenuation-only local-dose and raw-audit tests."""
import math

import pytest
import torch

from mechinterp.neuron_communities.nec03e_pairs import (
    PAIRS, attenuate_pair, make_pair, pair_dose_stats,
)
from mechinterp.neuron_communities.audit_nec03e import check_site, audit
from mechinterp.neuron_communities.source_sae import (
    SourceSAE, clean_activations,
)
from mechinterp.neuron_communities.matched_dose import graft_forward,physical_dose
from research.pef07.pef07 import Looper,as_tensor,source_split


def _donor(values):
    a=torch.zeros((len(values),64))
    a[:,0]=torch.tensor(values,dtype=a.dtype)
    return a


def test_always_attenuate_to_shared_physically_achievable_dose():
    W=torch.zeros((32,64))
    W[0,0]=1.
    left=_donor([1,2,0,3])
    right=_donor([2,1,1,6])
    x,y,receipt=attenuate_pair(left,right,W)
    dose_x=physical_dose(x,W)
    dose_y=physical_dose(y,W)
    target=torch.tensor([1.,1.,0.,3.])
    assert torch.allclose(dose_x,target,atol=1e-6)
    assert torch.allclose(dose_y,target,atol=1e-6)
    assert max(receipt["left_gain"])<=1
    assert max(receipt["right_gain"])<=1
    assert all(receipt["matched_within_1pct"])


def test_exact_raw_dose_contract_and_nonamplification():
    W=torch.zeros((32,64))
    W[0,0]=3.
    left=_donor([float(i+1) for i in range(16)])
    right=_donor([float(i+2) for i in range(16)])
    x,y,receipt=attenuate_pair(left,right,W)
    report=check_site(receipt)
    assert report["n"]==16
    assert report["matched"]==16
    assert report["active"]==16
    bad=dict(receipt)
    bad["left_gain"]=[float(x) for x in receipt["left_gain"]]
    bad["left_gain"][0]=1.1
    with pytest.raises(AssertionError,match="amplified|target dose"):
        check_site(bad)


def test_equal_dose_zero_control_transmits_no_signal():
    W=torch.zeros((32,64))
    W[0,0]=1.
    left=_donor([3.,1.])
    right=torch.zeros_like(left)
    a,b,rec=attenuate_pair(left,right,W)
    assert torch.count_nonzero(a)==0
    assert torch.count_nonzero(b)==0
    assert rec["shared_nonzero"]==[False,False]
    assert rec["retained_fraction_of_full_left_dose"]==[0.,0.]


def test_same_recurrent_site_graft_with_native_and_SAE_pairs():
    torch.set_num_threads(1)
    torch.manual_seed(103)
    model=Looper().eval()
    src,_=source_split()
    bits=as_tensor(src["train"][:16])
    act=clean_activations(model,bits)
    sae=SourceSAE().eval()
    mean=torch.zeros(64)
    scale=torch.ones(64)
    cand={"id":"synthetic","nodes":["0:1","5:3"]}
    ranking={
        "hub_disjoint_native_sham":["0:7","5:9"],
        "sae_feature_indices_by_visit":{"0":[0],"5":[1]},
        "sae_random_features_by_visit":{"0":[10],"5":[11]},
    }
    for pair in PAIRS:
        donors,doses,masks=make_pair(
            model,act,cand,ranking,["0:6","5:6"],sae,mean,scale,pair)
        assert set(donors)=={0,5}
        for mode in ("hubs","native","sham"):
            z=graft_forward(model,bits,donors,masks,mode)
            assert z.shape==(16,2) and torch.isfinite(z).all()
        stat=pair_dose_stats([doses])
        assert stat["site_sample_count"]==2*16
        assert stat["fraction_matched_within_1pct"] >= .99


def test_signal_gate_separate_from_dose_exactness():
    W=torch.zeros((32,64))
    W[0,0]=1.
    a,b,rec=attenuate_pair(_donor([5.]*16),_donor([0.1]*16),W)
    stat=pair_dose_stats([{0:rec}])
    assert stat["relative_dose_gate_pass"]
    assert not stat["minimum_signal_gate_pass"]
    assert stat["median_fraction_full_left_dose_preserved"]==pytest.approx(0.02,abs=1e-6)


def test_readonly_auditor_refuses_absent_upstream_archive(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit(tmp_path,[(103,"coalition")])
