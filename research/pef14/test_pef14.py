"""PEF-14 source partition, linearized phase mixture and projector invariants."""
import math
import torch
import pytest

from research.pef14 import pef14 as p
from research.pef07 import pef07 as p7


def test_fresh_source_partition_is_4096_complete_unique_contexts():
    sources,manifest=p.split_contexts()
    assert manifest['sha256']==p.sha_bytes(p.canonical({k:v for k,v in manifest.items() if k!='sha256'}))
    assert {k:len(v) for k,v in sources.items()} == {'train':2560,'probe':768,'heldout':768}
    old,_=(p7.split_histories() if hasattr(p7,'split_histories') else p7.source_split())
    # New SHA partition is independently defined and differs from PEF-07.
    assert tuple(map(tuple,sources['train'].tolist())) != old['train']
    seen=set()
    for name,x in sources.items():
        for group in x.reshape(-1,16,12):
            assert len({tuple(row[4:].tolist()) for row in group})==1
            assert [tuple(r[:4].tolist()) for r in group] == [tuple((m>>j)&1 for j in range(4)) for m in range(16)]
            for row in group:
                tup=tuple(row.tolist())
                assert tup not in seen
                seen.add(tup)
    assert len(seen)==4096


def test_projector_is_ungated_idempotent_and_removes_one_component():
    gen=torch.Generator().manual_seed(101)
    x=torch.randn((32,32),generator=gen)
    mu=torch.randn(32,generator=gen,dtype=torch.float64)
    g=p.unit(torch.randn(32,generator=gen,dtype=torch.float64))
    u=p.unit(torch.randn(32,generator=gen,dtype=torch.float64))
    u=p.unit(u-(u@g)*g)
    v,w=p.rotate(g,u,.25)
    assert abs(float(v@w))<1e-10
    y=p.project(x,mu,v)
    # Casting to float32 limits orthogonality to ~1e-6.
    assert torch.allclose((y.double()-mu)@v,torch.zeros(32,dtype=torch.float64),atol=3e-6)
    assert torch.allclose(p.project(y,mu,v),y,atol=3e-6)
    assert not torch.equal(x,y)


def test_phase_quantizer_bound_and_zero_step_failure():
    gen=torch.Generator().manual_seed(70)
    q=torch.randn(10000,generator=gen,dtype=torch.float64)*2
    for step in (.009,.07,1.5):
        d=p.quant_error(q,step)
        assert float(d.abs().max())<=step/2+1e-12
        assert torch.equal(p.quant_error(q,step),d)
    with pytest.raises(ValueError):p.quant_error(q,0)


def test_uniform_probability_handles_zero_derivative_and_threshold_tie():
    f=torch.tensor([0.,0.,1.,-1.,0.],dtype=torch.float64)
    j=torch.tensor([0.,0.,0.,0.,1.],dtype=torch.float64)
    y=torch.tensor([1,0,1,0,1])
    z=p.uniform_probability(f,j,y,0.4)
    assert torch.allclose(z,torch.tensor([1.,0.,1.,1.,.5],dtype=torch.float64))


def test_phase_mixture_is_context_permutation_invariant():
    gen=torch.Generator().manual_seed(77)
    f=torch.randn(768,generator=gen,dtype=torch.float64)
    j=torch.randn(768,generator=gen,dtype=torch.float64)
    q=torch.randn(768,generator=gen,dtype=torch.float64)
    y=(torch.randn(768,generator=gen)>0).long()
    a=p.phase_forecasts(f,j,y,q,.018)
    order=torch.randperm(48,generator=gen)
    def shuffle(z):return z.reshape(48,16)[order].reshape(768)
    b=p.phase_forecasts(*map(shuffle,(f,j,y,q)),.018)
    for name in ('distribution_mixture','conditional_cross','uniform_phase','exact_probe_phase','clean_probe_constant'):
        assert a[name] == pytest.approx(b[name],abs=1e-14)
    assert a['interval_90_low']==pytest.approx(b['interval_90_low'],abs=1e-14)


def test_conditional_independent_phase_population_bound_is_tight_example():
    # Joint law A=E in {0,1}: E[1[A=E]]=1; product law gives .5.
    joint={(0,0):.5,(1,1):.5}
    product={(a,e):.25 for a in (0,1) for e in (0,1)}
    tv=.5*sum(abs(joint.get((a,e),0)-product.get((a,e),0))
              for a in (0,1) for e in (0,1))
    risk_joint=sum(w for (a,e),w in joint.items() if a==e)
    risk_ind=sum(w for (a,e),w in product.items() if a==e)
    assert tv==.5 and abs(risk_joint-risk_ind)==tv


def test_pair_geometry_uses_hidden_states_only_and_fixed_seed():
    torch.manual_seed(123)
    h8=torch.randn(48*16,32)
    a=p.geometry(h8,211,'coalition')
    b=p.geometry(h8.clone(),211,'coalition')
    assert a['pc_scale']>0
    assert torch.equal(a['g'],b['g'])
    for key in p.TANGENTS:
        u=a['tangents'][key]
        assert abs(float(u@a['g']))<1e-9
        assert float(u.norm())==pytest.approx(1.,abs=1e-10)
        assert torch.equal(u,b['tangents'][key])
    assert not torch.equal(a['tangents']['random'],p.geometry(h8,223,'coalition')['tangents']['random'])


def test_linear_jacobian_is_correct_on_tiny_frozen_recurrence():
    torch.manual_seed(29)
    model=p.parent_model().eval()
    bits=torch.randint(0,2,(4,12))
    a=p.at_eight(model,bits)
    frozen=p.geometry(a.repeat(192,1),211,'coalition') if False else None
    w=p.unit(torch.randn(32,dtype=torch.float64))
    with torch.no_grad():
        final=p.after_eight(model,bits,a)
    fit=p7.fit_ridge(final,p.class_truth(bits,'coalition'))
    f,j=p.probe_margin_j(model,bits,a,fit,w)
    epsilon=1e-3
    with torch.no_grad():
        plus=p.ridge_margin(p.after_eight(model,bits,(a.double()+epsilon*w).float()),fit)
        minus=p.ridge_margin(p.after_eight(model,bits,(a.double()-epsilon*w).float()),fit)
    assert torch.allclose(j,(plus-minus)/(2*epsilon),atol=.01,rtol=.02)
    assert len(f)==len(j)==4
