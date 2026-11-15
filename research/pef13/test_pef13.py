"""PEF-13 mathematical invariants, lightweight and checkpoint-free."""
from __future__ import annotations
import math
import pytest
import torch

from research.pef11.pef11 import scalar_quant
from research.pef12.pef12 import uniform_phase_forecast
from research.pef13.pef13 import (
    STEPS, quant_residual, phase_forecast, phase_concentration,
    shuffled_seed, shuffled_forecast, correlation, canonical, decision_table,
)


def test_grid_is_prospective_and_disjoint_from_pef12():
    assert len(STEPS)==4 and len(set(STEPS))==4
    assert set(STEPS).isdisjoint({0.005,0.02,0.05})


def test_nearest_even_lattice_rounding_and_bounded_error():
    x=torch.tensor([-2.5,-1.5,-0.5,0.5,1.5,2.5,3.499999],dtype=torch.float64)
    err=quant_residual(x,1.)
    assert torch.allclose(x+err,torch.tensor([-2.,-2.,0.,0.,2.,2.,3.],dtype=torch.float64))
    assert bool((err.abs()<=.5+1e-12).all())
    with pytest.raises(ValueError): quant_residual(x,0.)


def test_phase_forecast_exact_linear_model():
    x=torch.tensor([-1.,-0.25,.2,1.1],dtype=torch.float64)
    q=torch.tensor([0.47,-0.44,0.57,0.61],dtype=torch.float64)
    j=torch.tensor([2.,1.2,-.8,.9],dtype=torch.float64)
    y=torch.tensor([0,0,1,1])
    d=quant_residual(q,1.)
    expectation=(((x+j*d)>=0).long()==y).double().mean().item()
    assert phase_forecast(x,j,y,q,1.)==expectation
    with pytest.raises(ValueError):phase_forecast(x,j,y[:3],q,1.)


def test_quantized_state_matches_scalar_lattice_residual():
    q=torch.tensor([.33,.66,-.49,-.52],dtype=torch.float64)
    w=torch.tensor([1.,0.],dtype=torch.float64)
    x=torch.stack((q,torch.tensor([3.,5.,7.,9.],dtype=torch.float64)),dim=1)
    quant=scalar_quant(x,torch.zeros(2,dtype=torch.float64),w,.25)
    d=quant_residual(q,.25)
    assert torch.allclose(quant[:,0]-x[:,0],d)
    assert torch.equal(quant[:,1],x[:,1])


def test_phase_concentration_and_shuffled_control():
    q=torch.arange(8,dtype=torch.float64)*0.25
    assert 0<=phase_concentration(q,1.)<1e-12
    assert abs(phase_concentration(torch.ones(10,dtype=torch.float64)*0.22,1.)-1)<1e-12
    seed=shuffled_seed(103,"coalition",0,1,2)
    d=quant_residual(torch.tensor([.1,.2,-.25,.4],dtype=torch.float64),.33)
    m=torch.tensor([-.05,.02,.1,.2],dtype=torch.float64)
    j=torch.tensor([1.,-1.,1.,-1.],dtype=torch.float64)
    t=torch.tensor([0,1,1,0])
    assert shuffled_forecast(m,j,t,d,seed)==shuffled_forecast(m,j,t,d,seed)
    with pytest.raises(ValueError):shuffled_seed(103,"coalition",2,0,0)


def test_uniform_phase_mathematics_and_corr_zero():
    m=torch.tensor([.1,-.2],dtype=torch.float64)
    j=torch.tensor([2.,4.],dtype=torch.float64)
    got=uniform_phase_forecast(m,j,.4)
    assert torch.allclose(got,torch.tensor([.625,.375],dtype=torch.float64))
    assert correlation(torch.ones(4),torch.arange(4))==0.


def test_conditional_taylor_curvature_bound():
    # f(x)=ax+b*x**2/2; the exact directional curvature is H=|b|.
    a,b,x=1.2,3.8,.4
    for delta in (-.31,-.11,.11,.31):
        f=lambda t:a*t+.5*b*t*t
        j=a+b*x
        mismatch=abs(f(x+delta)-f(x)-j*delta)
        assert mismatch<=abs(b)*delta*delta/2+1e-13
        assert math.isclose(mismatch,abs(b)*delta*delta/2,rel_tol=1e-12)


def test_registered_criterion_cannot_promote_failure():
    # A minimal synthetic frozen ledger and heldout table for decision aggregation.
    models=[]; held=[]
    for seed in (103,131,167):
        conds=[]; observed=[]
        for tangent in ("random","pair01"):
            for angle in (.25,4.):
                prior=[]; recorded=[]
                for rel in STEPS:
                    prior.append({"relative_step":rel,"phase_aware":.80,"uniform_phase":.9,
                                  "phase_shuffled":.85,"clean_constant":.9})
                    recorded.append({"relative_step":rel,"score":{"correct":int(.8*768)}})
                conds.append({"tangent":tangent,"angle_deg":angle,"steps":prior})
                observed.append({"tangent":tangent,"angle_deg":angle,"steps":recorded})
        models.append({"seed":seed,"task":"coalition","conditions":conds})
        held.append({"seed":seed,"task":"coalition","conditions":observed})
    verdict=decision_table({"models":models},held)
    assert verdict["primary_decision"]=="criterion_met"
    assert verdict["coalition_seeds_met"]==3
    assert len(canonical(verdict))>100