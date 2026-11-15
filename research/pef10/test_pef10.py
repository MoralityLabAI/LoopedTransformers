"""PEF-10 protocol invariants, executable with no original checkpoint ZIP."""
import math
import pytest
import torch

from research.pef10.pef10 import (
    EXPECTED_CONDITIONS, GRID, fixed_angles, identity_2d, rotate,
    tangent_pair, unit, stable_sign, TANGENTS,
)
from research.pef09.pef09 import projector


def test_fixed_grid_count_and_sign_identity():
    count = sum(len(fixed_angles(3.125 if name.startswith('pair') else 0, name))
                for name in TANGENTS)
    assert count == EXPECTED_CONDITIONS == 84
    assert GRID == (0, .25, .5, 1, 2, 4, 8, 16, 32, 64, 90)
    for name in TANGENTS:
        rows=fixed_angles(3.125 if name.startswith('pair') else 0,name)
        assert len(rows) == (22 if name.startswith('pair') else 20)
        assert len({(a,s) for a,s,_ in rows})==len(rows)
        assert sum(a==0 for a,s,_ in rows)==1
        assert sum(a==90 for a,s,_ in rows)==1
        assert all(s == 0 if a in (0,90) else s in (-1,1) for a,s,_ in rows)


def test_pair_anchor_is_original_pair_svd_line():
    g=torch.eye(32,dtype=torch.float64)[0]
    b=torch.eye(32,dtype=torch.float64)[1]
    for phi in (0.5,2,4,8,20,60):
        v=math.cos(math.radians(phi))*g+math.sin(math.radians(phi))*b
        u,angle,deg=tangent_pair(v,g,b)
        assert not deg
        assert abs(angle-phi)<1e-8
        assert abs(float(u@g))<1e-10
        assert torch.allclose(rotate(g,u,angle,1),v,atol=1e-12)
        assert torch.allclose(rotate(g,u,0,0),g)
        assert torch.allclose(rotate(g,u,90,0),b,atol=1e-12)
    with pytest.raises(ValueError):
        rotate(g,b,2,0)
    with pytest.raises(ValueError):
        rotate(g,b,0,1)


def test_exact_two_plane_projection_and_retained_energy_identity():
    e=torch.eye(32,dtype=torch.float64)
    g,u=e[0],e[1]
    gen=torch.Generator().manual_seed(90910)
    x=torch.randn(40,32,generator=gen,dtype=torch.float64)
    center=torch.randn(32,generator=gen,dtype=torch.float64)
    for deg in (0,.25,1,2,8,32,90):
        for sign in (-1,1):
            a,b=(x-center)@g,(x-center)@u
            theta=math.radians(deg)
            v=unit(math.cos(theta)*g+sign*math.sin(theta)*u)
            actual=(x-center)-(((x-center)@v)[:,None]*v[None,:])
            expected=torch.tensor([identity_2d(float(i),float(j),theta,sign)[0]
                                   for i,j in zip(a,b)],dtype=torch.float64)
            transverse=unit(-sign*math.sin(theta)*g+math.cos(theta)*u)
            residual=(x-center)-a[:,None]*g[None,:]-b[:,None]*u[None,:]
            assert torch.allclose(actual,expected[:,None]*transverse[None,:]+residual,atol=1e-11)
            assert torch.allclose(actual @ v,torch.zeros(len(x),dtype=x.dtype),atol=1e-11)
            assert torch.allclose(actual+center,projector(x,center,v),atol=1e-11)


def test_ungated_projector_is_permutation_equivariant():
    g=torch.eye(32,dtype=torch.float64)[0]
    u=torch.eye(32,dtype=torch.float64)[1]
    theta=rotate(g,u,4,1)
    x=torch.randn(48,32,generator=torch.Generator().manual_seed(100),dtype=torch.float64)
    center=torch.randn(32,dtype=torch.float64)
    permutation=torch.randperm(len(x),generator=torch.Generator().manual_seed(2))
    assert torch.equal(projector(x[permutation],center,theta),projector(x,center,theta)[permutation])
    assert torch.equal(projector(center.unsqueeze(0),center,theta),center.unsqueeze(0))


def test_frozen_direction_is_orthogonal_and_degenerate_fallback_is_named():
    g=torch.eye(32,dtype=torch.float64)[0]
    fallback=torch.eye(32,dtype=torch.float64)[1]
    u,phi,deg=tangent_pair(g,g,fallback)
    assert deg and phi==0
    assert torch.equal(u,fallback)
    v=-unit(g*2+fallback)
    w,angle,deg=tangent_pair(v,g,fallback)
    assert not deg
    assert float(w @ fallback)>0
    assert angle>0
    assert float(w@g)<1e-12
