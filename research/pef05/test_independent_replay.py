"""PEF-05 independent source-set and mathematical null checks."""
import torch
from research.pef05.independent_replay import _independent_source_split


def test_pair_closed_source_is_reconstructed_independently():
    s=_independent_source_split()
    assert {k:len(v) for k,v in s.items()}=={
        "train":160,"probe":48,"heldout":48
    }
    full=torch.cat(list(s.values()),dim=0)
    assert len({tuple(v) for v in full.tolist()})==256
    for rows in s.values():
        assert torch.all(rows[0::2,0]==0)
        assert torch.all(rows[1::2,0]==1)
        assert torch.equal(rows[0::2,1:],rows[1::2,1:])


def test_every_suffix_only_prediction_has_exact_half_accuracy_and_no_joint_pairs():
    test=_independent_source_split()["heldout"]
    for j in range(1,8):
        prediction=test[:,j]
        assert int((prediction==test[:,0]).sum())==24
        assert int(((prediction[0::2]==0)&(prediction[1::2]==1)).sum())==0
