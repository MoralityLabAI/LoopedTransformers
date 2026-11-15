"""PEF-15 split-conformal and finite-population bounds for complete contexts."""
from itertools import combinations
import math

import pytest
import torch
from research.pef15 import pef15 as p
from research.pef14 import pef14 as parent


def test_complete_disjoint_new_160_32_16_48_context_groups():
    groups,manifest=p.split_contexts()
    assert manifest['sha256']==parent.sha_bytes(p.canonical({k:v for k,v in manifest.items() if k!='sha256'}))
    assert {k:len(x) for k,x in groups.items()}=={
        'train':2560,'fit':512,'calibration':256,'heldout':768}
    words={}
    for name,rows in groups.items():
        for cohort in rows.reshape(-1,16,12):
            c=tuple(cohort[0,4:].tolist())
            assert c not in words
            words[c]=name
            assert all(tuple(x[4:].tolist())==c for x in cohort)
            assert [tuple(x[:4].tolist()) for x in cohort] == [
                tuple((mask>>k)&1 for k in range(4)) for mask in range(16)]
    assert len(words)==256
    previous,_=parent.split_contexts()
    assert not torch.equal(groups['train'],previous['train'])


def test_registered_conformal_rank_and_calibration_example():
    assert p.conformal_rank(16,.1)==16
    assert p.conformal_rank(19,.1)==18
    a=p.interval_from_calibration(.7,[.7]*8+[.8]*8)
    assert a['rank']==16 and a['low']==pytest.approx(.6)
    assert a['high']==pytest.approx(.8)
    assert p.interval_from_calibration(.7,[.0]*16)['low']==0
    assert p.interval_from_calibration(.7,[.0]*16)['high']==1
    with pytest.raises(ValueError): p.interval_from_calibration(.9,[])


def test_exact_exchangeable_context_rank_coverage():
    # For 17 distinct exchangeable context rates, the new item is excluded
    # only when it is the largest absolute residual among 17 values.
    point=.5
    scores=[(x+1)/100 for x in range(17)]
    hits=[]
    for candidate in range(17):
        held=scores[candidate]
        calibration=[s for i,s in enumerate(scores) if i!=candidate]
        r=p.interval_from_calibration(point,[point+s for s in calibration])
        hits.append(held <= r['radius']+1e-12)
    assert sum(hits)==16 and sum(hits)/17> .9


def test_serfling_radius_matches_frozen_finite_population_formula():
    rates=[.75]*16
    a=p.serfling_heldout_interval(rates)
    expect=(64/48)*math.sqrt((1-15/64)*math.log(20)/(32))
    assert a['radius']==pytest.approx(expect,rel=1e-12)
    assert a['center']==.75 and a['high']==1
    with pytest.raises(ValueError):p.serfling_heldout_interval([1.01])


def test_conditional_mixture_is_focal_context_permutation_invariant():
    g=torch.Generator().manual_seed(415)
    f=torch.randn(512,dtype=torch.float64,generator=g)
    j=torch.randn(512,dtype=torch.float64,generator=g)
    q=torch.randn(512,dtype=torch.float64,generator=g)
    y=torch.randint(2,(512,),generator=g)
    a=p.phase_forecast(f,j,y,q,.075)
    order=torch.randperm(32,generator=g)
    reshape=lambda z:z.reshape(32,16)[order].reshape(512)
    b=p.phase_forecast(reshape(f),reshape(j),reshape(y),reshape(q),.075)
    for k in a:
        assert a[k]==pytest.approx(b[k],abs=1e-12)
    assert 0 <= a['distribution_mixture'] <= 1


def test_linear_phase_pseudocount_formula_and_tie_cases():
    # When J=0, all phase transformations leave the class unchanged.
    f=torch.tensor([2.]*256+[-2.]*256,dtype=torch.float64)
    j=torch.zeros(512,dtype=torch.float64)
    q=torch.zeros(512,dtype=torch.float64)
    y=torch.tensor([1]*256+[0]*256)
    out=p.phase_forecast(f,j,y,q,0.04)
    assert all(v==pytest.approx(1.0) for v in out.values())


def test_new_training_protocol_does_not_reuse_old_seeds_or_conditions():
    assert set(p.SEEDS).isdisjoint(parent.SEEDS)
    assert set(p.STEPS).isdisjoint(parent.STEPS)
    assert len(p.SEEDS)*len(p.TASKS)*len(p.TANGENTS)*len(p.ANGLES)*len(p.STEPS)==96
    assert p.COUNTS['calibration']+p.COUNTS['heldout']==64


def test_frozen_ridge_and_projected_memories_are_natural():
    torch.manual_seed(51)
    model=parent.parent_model().eval()
    words,_=p.split_contexts()
    fit=words['fit'][:32]
    with torch.no_grad():
        h8=parent.at_eight(model,fit)
        manual=parent.after_eight(model,fit,h8)
        assert torch.equal(manual,model.memory(fit))
        ridge=parent.p7.fit_ridge(manual,parent.class_truth(fit,'coalition'))
        margin=parent.ridge_margin(manual,ridge)
        assert torch.equal((margin>=0).long(),parent.p7.apply_ridge(manual,ridge))
