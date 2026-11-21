"""NEC-03J native 64D/32D Jacobian and transport unit tests on CPU."""
from __future__ import annotations

import pytest
import torch

from mechinterp.neuron_communities.nec03j_direction_transport import (
    FD_MAX_ABSOLUTE_ERROR, JACOBIAN_IDENTITY_MAX_ERROR,
    oriented_margin, traced_forward, finite_difference_check, analyze_pair,
)
from research.pef07.pef07 import Looper,source_split,as_tensor,model_digest


def _model_and_bits():
    torch.set_num_threads(1)
    torch.manual_seed(103)
    model=Looper().eval()
    catalog,_=source_split()
    bits=as_tensor(catalog["train"][:16])
    return model,bits,catalog


def test_gradients_are_native_64d_and_mlpdown_32d_for_all_twelve_visits():
    model,bits,_=_model_and_bits()
    sha=model_digest(model)
    result=traced_forward(model,bits,mode="clean",derivative=True)
    assert result["logits"].shape==(16,2)
    assert result["memory"].shape==(16,12,32)
    assert len(result["grad_gelu"])==len(result["grad_down"])==12
    W=model.block.mlp_down.weight.detach()
    for visit in range(12):
        g=result["grad_gelu"][visit]
        out=result["grad_down"][visit]
        assert g.shape==(16,64)
        assert out.shape==(16,32)
        assert torch.isfinite(g).all() and torch.isfinite(out).all()
        assert torch.allclose(g,out@W,atol=1e-5,rtol=1e-4)
    assert model_digest(model)==sha


def test_clean_orientation_is_identical_across_modified_cases():
    model,bits,_=_model_and_bits()
    clean=traced_forward(model,bits,mode="clean")
    chosen=clean["logits"].argmax(1)
    clean_margin=oriented_margin(clean["logits"],chosen)
    assert torch.all(clean_margin>=0)
    with pytest.raises(ValueError):
        oriented_margin(clean["logits"][:15],chosen)


def test_source_only_finite_difference_matches_autograd():
    model,bits,catalog=_model_and_bits()
    check=finite_difference_check(
        model,catalog,{"nodes":["2:4"],"id":"source-only-test"})
    assert check["max_absolute_fd_error"]<=FD_MAX_ABSOLUTE_ERROR
    assert check["source_split"]=="train[:16] only"
    assert check["original_16_source_histories_only"]


def test_same_hubs_only_recurrence_and_nonlinear_replay_memory_drift():
    model,bits,_=_model_and_bits()
    clean=traced_forward(model,bits,mode="clean",derivative=True)
    pred=clean["logits"].argmax(1)
    nodes={2:torch.zeros(16,64),7:torch.zeros(16,64)}
    for visit in nodes:
        nodes[visit][:,3]=0.2
    donors={}
    gates={}
    doses={}
    W=model.block.mlp_down.weight.detach()
    for visit,source in nodes.items():
        control=source*0.5
        donors[visit]={"native":source,"sham":control,
                       "hubs":torch.zeros_like(source)}
        mask=torch.zeros(64)
        mask[5]=1.
        gates[visit]=mask
        denom=(source @ W.T).norm(dim=1)
        doses[visit]={
            "full_left_dose":denom.tolist(),
            "full_right_dose":(control @ W.T).norm(dim=1).tolist(),
            "shared_achievable_dose":(control @ W.T).norm(dim=1).tolist(),
            "left_gain":[1.]*16,
            "right_gain":[1.]*16}
    hubs=traced_forward(model,bits,mode="hubs",donors=donors,gates=gates,
                        orientation=pred,derivative=True)
    left=traced_forward(model,bits,mode="native",donors=donors,gates=gates,
                        orientation=pred)
    right=traced_forward(model,bits,mode="sham",donors=donors,gates=gates,
                         orientation=pred)
    effect=analyze_pair(model,clean,hubs,left,right,donors,doses)
    assert len(effect["hubs_linear_margin_delta"])==16
    assert len(effect["nonlinear_exact_margin_delta"])==16
    assert len(effect["source_vs_control_memory_drift_by_visit"])==16
    assert all(len(row)==12 for row in effect["source_vs_control_memory_drift_by_visit"])
    assert set(effect["local_jacobian"])=={"2","7"}
    assert effect["max_native_gelu_to_mlpdown_jacobian_discrepancy"]<=JACOBIAN_IDENTITY_MAX_ERROR
    assert all(torch.isfinite(torch.tensor(row)).all()
               for row in effect["source_vs_control_memory_drift_by_visit"])
    assert effect["mean_absolute_nonlinear_remainder"]>=0


def test_confirmation_refusal_on_wrong_dimensions():
    model,bits,_=_model_and_bits()
    with pytest.raises(ValueError):
        traced_forward(model,bits[:,:11],mode="clean")
    with pytest.raises(ValueError):
        traced_forward(model,bits,mode="unknown")
    with pytest.raises(ValueError):
        traced_forward(model,bits,mode="hubs")
