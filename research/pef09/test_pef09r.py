"""PEF-09R orthogonal residualization and fixed ridge recovery tests."""
import torch
import pytest
from research.pef09.pef09r import rank_project, pca_geometry, fit_for_states, predict


def test_projection_removes_rank_k_spans_without_gating():
    rng=torch.Generator().manual_seed(17)
    x=torch.randn((48,32),generator=rng,dtype=torch.float64)
    center,axes,summary=pca_geometry(x)
    assert (axes@axes.T-torch.eye(32,dtype=torch.float64)).abs().max()<1e-9
    for k in (1,2,4,8):
        out=rank_project(x,center,axes[:k])
        assert torch.allclose((out-center)@axes[:k].T,
                              torch.zeros(48,k,dtype=torch.float64),atol=1e-10)
        assert torch.allclose(rank_project(out,center,axes[:k]),out,atol=1e-10)
    assert summary['top_k_variance_fraction']['1']>0
    assert summary['top_k_variance_fraction']['8']<=1


def test_recovery_fits_only_supplied_probe_states():
    torch.manual_seed(33)
    # The observed coordinates can be erased while a residual label feature survives.
    g=torch.Generator().manual_seed(71)
    p=torch.randint(0,2,(128,),generator=g).long()
    states=torch.randn(128,32,generator=g)
    states[:,0]=p.float()*20-10
    states[:,1]=p.float()*2-1
    center=states.mean(0)
    residual=rank_project(states,center,torch.eye(32)[:1])
    assert residual[:,0].allclose(center[0].expand(128),atol=1e-6)
    fit=fit_for_states(residual,p)
    pred=predict(residual,fit)
    assert int((pred==p).sum())==128


def test_bad_axes_are_rejected():
    with pytest.raises(AssertionError):
        rank_project(torch.ones((3,32)),torch.zeros(32),torch.ones((1,32)))
    with pytest.raises(ValueError):
        rank_project(torch.ones((3,32)),torch.zeros(32),torch.ones((2,31)))
