"""PEF-11 mathematical and serialization invariants; no checkpoint required."""
import math
import pytest
import torch

from research.pef11.pef11 import (
    ANGLES, TANGENTS, NOISE_REL, QUANT_REL, REPLICATES,
    offplane, orientation, gaussian_z, scalar_quant, channel_quant,
    perturb_gaussian, expected_noise_seed, select_20,
)


def test_fixed_grid_and_replicates_are_exact():
    assert ANGLES == (0, .25, 1, 4, 16)
    assert TANGENTS == ("pair01", "pair23", "pca2", "random")
    assert NOISE_REL == (0, .001, .003, .01, .03, .1)
    assert QUANT_REL == (0, .003, .01, .03, .1)
    assert REPLICATES == (0, 1, 2)
    assert len(ANGLES)*len(TANGENTS)==20


def test_exact_angle_projection_and_tangent_retention():
    g,u=torch.eye(32,dtype=torch.float64)[:2]
    for angle in ANGLES:
        v,w=orientation(g,u,angle)
        assert torch.allclose(torch.stack((v,w))@torch.stack((v,w)).T,
                              torch.eye(2,dtype=torch.float64),atol=1e-12)
        a,b=3.1,-1.7
        x=a*g+b*u
        post=x-(x@v)*v
        q=-a*math.sin(math.radians(angle))+b*math.cos(math.radians(angle))
        assert torch.allclose(post,q*w,atol=1e-11)
        assert abs(float(post@v))<1e-11


def test_offplane_unit_fixed_and_equal_energy_noise():
    g,u=torch.eye(32,dtype=torch.float64)[:2]
    o=offplane(g,u,102030)
    assert torch.allclose(o,offplane(g,u,102030))
    assert float(o.norm())==pytest.approx(1,abs=1e-12)
    assert abs(float(o@g))+abs(float(o@u))<1e-12
    _,w=orientation(g,u,4)
    x=torch.randn(16,32,generator=torch.Generator().manual_seed(11))
    noise=gaussian_z(120,16)
    for sig in (.01,.1):
        transverse=perturb_gaussian(x,w,noise,sig)
        outside=perturb_gaussian(x,o,noise,sig)
        assert torch.allclose((transverse-x).norm(dim=1),(outside-x).norm(dim=1),atol=2e-6)


def test_quantization_uses_probe_fixed_coordinate_and_is_idempotent_in_double():
    e=torch.eye(32,dtype=torch.float64)
    center=1.5*e[3]
    w=e[0]
    x=torch.randn(33,32,generator=torch.Generator().manual_seed(4),dtype=torch.float64)
    y=scalar_quant(x,center,w,.1)
    assert torch.allclose((y-center)@w/.1,torch.round((x-center)@w/.1),atol=1e-11)
    residual=y-x
    assert torch.allclose(residual-residual[:,0,None]*w[None,:],torch.zeros_like(residual),atol=1e-12)
    assert torch.equal(scalar_quant(x,center,w,0),x)
    assert torch.equal(scalar_quant(y,center,w,.1),y)
    with pytest.raises(ValueError):
        scalar_quant(x,center,w,-.1)


def test_channel_quantization_centered_and_nearest_even():
    center=torch.zeros(32,dtype=torch.float64)
    step=torch.ones(32,dtype=torch.float64)*.25
    x=torch.ones(4,32,dtype=torch.float64)*.34
    q=channel_quant(x,center,step)
    assert torch.allclose(q,torch.ones_like(q)*.25)
    assert torch.equal(channel_quant(x,center,torch.zeros_like(step)),x)
    with pytest.raises(ValueError):
        channel_quant(x,center,torch.cat((torch.tensor([-1.0]),step[1:])))
    test=torch.tensor([-.5,-1.5,.5,1.5],dtype=torch.float64).unsqueeze(1).expand(-1,32)*step
    nearest=channel_quant(test,center,step)
    assert [int(round(float(q / .25))) for q in nearest[:,0]]==[0,-2,0,2]


def test_noise_seeds_reproducible_and_unique_on_registered_grid():
    seen=[]
    for task in ('coalition','threshold'):
        for seed in (103,131,167):
            for t in range(4):
                for a in range(5):
                    for rep in REPLICATES:
                        k=expected_noise_seed(seed,task,t,a,rep)
                        seen.append(k)
                        assert torch.equal(gaussian_z(k,16),gaussian_z(k,16))
    assert len(set(seen))==len(seen)


def test_ungated_projection_and_quantization_permutation_equivariance():
    g,u=torch.eye(32,dtype=torch.float64)[:2]
    v,w=orientation(g,u,1)
    center=torch.randn(32,generator=torch.Generator().manual_seed(7),dtype=torch.float64)
    x=torch.randn(32,32,generator=torch.Generator().manual_seed(8),dtype=torch.float64)
    perm=torch.randperm(32,generator=torch.Generator().manual_seed(9))
    projection=lambda m: m-((m-center)@v)[:,None]*v[None,:]
    y=projection(x)
    assert torch.equal(projection(x[perm]),y[perm])
    assert torch.equal(scalar_quant(y[perm],center,w,.1),scalar_quant(y,center,w,.1)[perm])
    step=torch.arange(1,33,dtype=torch.float64)*.001
    assert torch.equal(channel_quant(y[perm],center,step),channel_quant(y,center,step)[perm])
