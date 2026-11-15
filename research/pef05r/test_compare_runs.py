"""PEF-05R independent tensor drift and optimizer-moment comparison tests."""
import torch
import pytest
from research.pef05r.compare_runs import drift, optimizer_moments, parameter_vector


def test_equal_model_vectors_have_zero_drift():
    x={"head":torch.tensor([1.,-2.]),"bias":torch.tensor([0.])}
    assert parameter_vector(x).tolist()==[0.,1.,-2.]
    p=parameter_vector(x)
    result=drift(p,p.clone())
    assert result["relative_l2"]==0
    assert result["exactly_equal"]
    assert result["maximum_absolute_element"]==0


def test_nonzero_drift_is_normalized_to_both_norms():
    x=torch.tensor([3.,4.])
    y=torch.tensor([0.,4.])
    result=drift(x,y)
    assert result["absolute_l2"]==pytest.approx(3.0)
    assert result["relative_l2"]==pytest.approx(3.0/4.5)
    assert not result["exactly_equal"]
    with pytest.raises(AssertionError):
        drift(x,torch.tensor([float("nan"),3.]))


def test_adamw_moments_gather_in_parameter_index_order():
    obj={"state":{
       5:{"exp_avg":torch.tensor([-2.]),"exp_avg_sq":torch.tensor([7.])},
       2:{"exp_avg":torch.tensor([1.,2.]),"exp_avg_sq":torch.tensor([3.,4.])},
    }}
    assert optimizer_moments(obj,"exp_avg").tolist()==[1.,2.,-2.]
    assert optimizer_moments(obj,"exp_avg_sq").tolist()==[3.,4.,7.]
    with pytest.raises(AssertionError):
        optimizer_moments(obj,"step")
