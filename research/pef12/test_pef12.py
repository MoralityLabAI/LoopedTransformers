"""PEF-12 algebra, calibration, deterministic seeding and frozen-weight invariants."""
from __future__ import annotations

import math

import pytest
import torch

from research.pef07 import pef07 as p7
from research.pef08 import pef08 as p8
from research.pef09 import pef09 as p9
from research.pef10 import pef10 as p10
from research.pef11 import pef11 as p11
from research.pef12 import pef12 as p12


def test_gaussian_probability_is_exact_for_linear_margin():
    m = torch.tensor([0., 1., -1., 10., -10., 0.], dtype=torch.float64)
    d = torch.tensor([1., 1., 1., 1., 1., 0.], dtype=torch.float64)
    answer = p12.exact_linearized_gaussian(m,d,1.)
    assert answer[0] == .5 and answer[5] == .5
    assert math.isclose(float(answer[1]),.8413447460685429,rel_tol=1e-9)
    assert math.isclose(float(answer[2]),1-.8413447460685429,rel_tol=1e-9)
    assert answer[3]>0.9999 and answer[4]<.0001
    assert torch.allclose(answer[:3],p12.exact_linearized_gaussian(m[:3],-d[:3],1.))
    with pytest.raises(ValueError):
        p12.exact_linearized_gaussian(m,d,0)


def test_zero_jacobian_limiting_correctness_probabilities():
    m=torch.tensor([-1.,0.,1.])
    d=torch.zeros(3)
    assert p12.exact_linearized_gaussian(m,d,.01).tolist()==[0.,.5,1.]
    assert p12.uniform_phase_forecast(m,d,.01).tolist()==[0.,.5,1.]


def test_uniform_phase_approximation_and_nearest_even_lattice():
    m=torch.tensor([-0.5,-0.25,0.,0.25,0.5],dtype=torch.float64)
    d=torch.ones(5,dtype=torch.float64)
    assert torch.allclose(p12.uniform_phase_forecast(m,d,1),
                          torch.tensor([0,.25,.5,.75,1],dtype=torch.float64))
    with pytest.raises(ValueError):
        p12.uniform_phase_forecast(m,d,0)
    torch.manual_seed(11)
    states=torch.randn(160,32)
    center=torch.zeros(32)
    w=torch.zeros(32);w[0]=1
    step=.3
    quant=p11.scalar_quant(states,center,w,step)
    assert float((quant-states).double().norm(dim=1).max()) <= step/2 + 1e-6
    assert torch.equal(p11.scalar_quant(states,center,w,0),states)


def test_gaussian_probability_matches_monte_carlo_for_fixed_linear_map():
    sigma=0.6
    margin=torch.tensor([.2, -.4, 1.1],dtype=torch.float64)
    derivative=torch.tensor([2.,-1.,.4],dtype=torch.float64)
    predicted=p12.exact_linearized_gaussian(margin,derivative,sigma)
    z=torch.randn(3,200_000,dtype=torch.float64,generator=torch.Generator().manual_seed(202))
    empirical=((margin[:,None]+sigma*derivative[:,None]*z)>=0).double().mean(dim=1)
    assert torch.allclose(predicted,empirical,atol=.004)


def test_noise_seed_schedule_unique_and_reproducible():
    seeds=[p12.noise_seed(seed,task,t,a,level,rep) for seed in (103,131,167)
            for task in ("coalition","threshold") for t in range(2)
            for a in range(2) for level in range(3) for rep in range(p12.REPS)]
    assert len(seeds)==6*2*2*3*16
    assert len(seeds)==len(set(seeds))
    with pytest.raises(ValueError):
        p12.noise_seed(103,"coalition",2,0,0,0)
    left=p11.gaussian_z(seeds[0],768)
    assert torch.equal(left,p11.gaussian_z(seeds[0],768))


def test_probe_input_jacobian_matches_centered_finite_difference():
    torch.manual_seed(1234)
    model=p7.CoalitionMemoryLooper().eval()
    catalog,_,_=p8.source_data()
    bits=catalog["probe"][:12]
    initial=p8.states_by_loop(model,bits)[8]
    ridge=p7.fit_ridge(p8.states_by_loop(model,catalog["probe"][:128])[12],
                       p8.labels(catalog["probe"][:128],"coalition"))
    w=torch.randn(32,dtype=torch.float64)
    w=w/w.norm()
    original=p8.weights_digest(model)
    margins,j=p12.probe_margin_jacobians(model,bits,initial,ridge,{"axis":w})
    eps=0.01
    with torch.no_grad():
        plus=p8.ridge_margin(p12.exact_suffix_h12(model,bits,
            initial + eps*w.float()),ridge)
        minus=p8.ridge_margin(p12.exact_suffix_h12(model,bits,
            initial - eps*w.float()),ridge)
    approx=(plus-minus)/(2*eps)
    assert torch.allclose(j["axis"],approx,atol=.15,rtol=.04)
    assert len(margins)==len(bits)
    assert p8.weights_digest(model)==original


def test_orthogonal_rotation_and_preregistered_grid_sizes():
    gen=torch.Generator().manual_seed(24680)
    q,_=torch.linalg.qr(torch.randn((32,2),dtype=torch.float64,generator=gen))
    g,u=q[:,0],q[:,1]
    for a in p12.ANGLES:
        v,w=p11.orientation(g,u,a)
        assert abs(float(v@w))<1e-9
        assert abs(float(v@v)-1)<1e-9
        x=torch.randn(10,32,dtype=torch.float64,generator=gen)
        xproj=x-(x@v)[:,None]*v[None,:]
        assert torch.allclose(xproj@v,torch.zeros(10,dtype=torch.float64),atol=1e-12)
    assert len(p12.TANGENTS)*len(p12.ANGLES)*6==24
    assert len(p12.GAUSS)==len(p12.QUANT)==3


def test_independent_standard_library_forecast_agrees_with_torch():
    from research.pef12.audit_pef12 import forecast
    gen=torch.Generator().manual_seed(2030)
    margin=torch.randn(64,dtype=torch.float64,generator=gen)
    deriv=torch.randn(64,dtype=torch.float64,generator=gen)
    scale=1.7
    for rel in p12.GAUSS:
        a=float(p12.exact_linearized_gaussian(margin,deriv,scale*rel).mean())
        b=forecast(margin.tolist(),deriv.tolist(),scale,rel,"gauss")
        assert abs(a-b)<1e-12
    for rel in p12.QUANT:
        a=float(p12.uniform_phase_forecast(margin,deriv,scale*rel).mean())
        b=forecast(margin.tolist(),deriv.tolist(),scale,rel,"quant")
        assert abs(a-b)<1e-12


def test_taylor_disagreement_requires_small_linearized_margin():
    # f(x)=x+x² has |f''|=2, so |nonlinear-linear|=|epsilon|².
    for x in [-.8,-.1,0.,.3,.7]:
        for epsilon in [-.9,-.4,-.05,0.,.07,.5,.8]:
            actual=(x+epsilon)+(x+epsilon)**2
            linear=x+x*x+epsilon*(1+2*x)
            assert abs(actual-linear)<=epsilon**2+1e-12
            if (actual>=0)!=(linear>=0):
                assert abs(linear)<=epsilon**2+1e-12
