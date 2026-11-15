"""PEF-09 mask-independent projector and source-only selection invariants."""
import hashlib
import torch
import pytest
from research.pef09.pef09 import (
    DIM, PAIRS, LOOPS, all_base_differences, leading_direction,
    projector, projector_many, random_energy_matches, projection_energy,
    select_probe_features, replay_batched,
)
from research.pef08 import pef08 as p8
from research.pef07 import pef07 as p7


def synthetic_cube(num_contexts=3):
    states=torch.zeros(num_contexts,16,DIM,dtype=torch.float64)
    for c in range(num_contexts):
        for mask in range(16):
            a,b,d,e=[(mask>>i)&1 for i in range(4)]
            states[c,mask,0]=3*a*b+0.4*c
            states[c,mask,1]=5*d*e-0.3*c
            states[c,mask,2]=2*a+4*b+3*d+e
    return states.reshape(-1,DIM)


def test_all_four_other_bit_bases_yield_separate_pair_signals():
    state=synthetic_cube()
    a=all_base_differences(state,(0,1))
    b=all_base_differences(state,(2,3))
    assert a.shape==(12,DIM)
    assert b.shape==(12,DIM)
    assert torch.allclose(a[:,0],torch.full((12,),3.,dtype=torch.float64))
    assert torch.allclose(b[:,1],torch.full((12,),5.,dtype=torch.float64))
    assert torch.count_nonzero(a[:,1:])==0
    assert torch.count_nonzero(b[:,:1])==0
    assert torch.count_nonzero(b[:,2:])==0
    with pytest.raises(ValueError):
        all_base_differences(state,(1,1))


def test_label_free_svd_recovers_two_distinct_causal_directions():
    state=synthetic_cube()
    first,_,deg0=leading_direction(all_base_differences(state,(0,1)))
    second,_,deg1=leading_direction(all_base_differences(state,(2,3)))
    assert not (deg0 or deg1)
    assert torch.allclose(first,torch.eye(DIM,dtype=torch.float64)[0],atol=1e-12)
    assert torch.allclose(second,torch.eye(DIM,dtype=torch.float64)[1],atol=1e-12)
    assert torch.isclose(first@second,torch.tensor(0.,dtype=torch.float64))


def test_ungated_projection_applies_to_zero_and_single_focal_masks():
    states=synthetic_cube()
    mean=states.mean(0)
    first,_sv,_degen=leading_direction(all_base_differences(states,(0,1)))
    out=projector(states,mean,first)
    # The pair interaction is excised by a GLOBAL operator for all source masks.
    assert float(all_base_differences(out,(0,1)).abs().max())<1e-12
    assert torch.allclose(all_base_differences(out,(2,3)),
                          all_base_differences(states,(2,3)),atol=1e-12)
    original_zero=states[0,0]
    new_zero=out[0,0]
    assert not torch.equal(original_zero,new_zero)
    projected=projector_many(states.float(),mean,first[None,:])
    assert torch.allclose(projected[0],out.float(),atol=1e-6)


def test_projector_algebra_and_full_space_collapse():
    g=torch.Generator().manual_seed(17)
    state=torch.randn(5,DIM,generator=g,dtype=torch.float64)
    mean=torch.randn(DIM,generator=g,dtype=torch.float64)
    u=torch.randn(DIM,generator=g,dtype=torch.float64)
    u=u/u.norm()
    patch=projector(state,mean,u)
    assert torch.allclose(projector(patch,mean,u),patch,atol=1e-12)
    assert torch.allclose(state-patch,((state-mean)@u)[:,None]*u[None,:])
    full=state.clone()
    for axis in torch.eye(DIM,dtype=torch.float64):
        full=projector(full,mean,axis)
    assert torch.allclose(full,mean.expand_as(state),atol=1e-12)
    with pytest.raises(AssertionError):
        projector(state,mean,2*u)


def test_energy_matched_nulls_are_deterministic_and_probe_only():
    gen=torch.Generator().manual_seed(2026)
    probe=torch.randn(768,DIM,dtype=torch.float64,generator=gen)
    mean=probe.mean(0)
    target=torch.eye(DIM,dtype=torch.float64)[0]
    a=random_energy_matches(probe,mean,target,seed=103,task='coalition',loop=8,pair_idx=0)
    b=random_energy_matches(probe.clone(),mean.clone(),target,seed=103,task='coalition',loop=8,pair_idx=0)
    assert a==b and len(a)==8
    assert len({v['candidate_index'] for v in a})==8
    assert all(abs(torch.tensor(x['unit']).norm().item()-1)<1e-12 for x in a)
    assert all(x['probe_projection_energy']>0 for x in a)
    alt=random_energy_matches(probe,mean,target,seed=103,task='threshold',loop=8,pair_idx=0)
    assert [x['candidate_index'] for x in a]!=[x['candidate_index'] for x in alt]


def test_batched_natural_replay_exactly_matches_individual_replay():
    torch.manual_seed(103)
    cls = p7.Looper if hasattr(p7,'Looper') else p7.CoalitionMemoryLooper
    model=cls().eval()
    catalog,_,_=p8.source_data()
    bits=catalog['probe'][:32]
    with torch.no_grad():
        trace=p8.states_by_loop(model,bits)
        for visit in LOOPS:
            original=trace[visit]
            dirs=torch.eye(DIM)[:2]
            variations=projector_many(original,original.mean(0).double(),dirs)
            actual_h12,actual_h14=replay_batched(model,bits,variations,visit)
            for idx in range(2):
                expected_h12=p8.replay(model,bits,variations[idx],visit,12)
                expected_h14=p8.replay(model,bits,variations[idx],visit,14)
                assert torch.equal(actual_h12[idx],expected_h12)
                assert torch.equal(actual_h14[idx],expected_h14)
