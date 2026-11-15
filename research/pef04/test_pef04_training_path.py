"""PEF-04C frozen schedule and isolated read-only intermediate snapshot tests."""
import torch
from research.pef03.pef03 import BitMemoryLooper,tensor,hash_weights
from research.pef04.pef04 import paired_split
from research.pef04.pef04_training_path import SCAN_STEPS, TRAIN_STEPS, checkpoint_metrics


def test_retrospective_sampling_grid_is_fixed_and_bounded():
    assert SCAN_STEPS == (0,10,25,50,75,100,125,150,175,200,225,250)
    assert SCAN_STEPS[0]==0 and SCAN_STEPS[-1]==TRAIN_STEPS
    assert list(SCAN_STEPS)==sorted(set(SCAN_STEPS))


def test_snapshot_does_not_modify_seeded_model():
    torch.set_num_threads(1)
    torch.manual_seed(157)
    model=BitMemoryLooper().train()
    before=hash_weights(model)
    catalog,_=paired_split()
    row=checkpoint_metrics(model,tensor(catalog["probe"]),
                           tensor(catalog["heldout"]),0)
    assert row["optimizer_step"]==0
    assert row["original_task_correct_out_of_48"] in range(49)
    assert row["oldest_bit_correct_out_of_48"] in range(49)
    assert row["both_members_correct_pairs_out_of_24"] in range(25)
    assert hash_weights(model)==before


def test_pair_closed_snapshot_expected_sizes():
    c,_=paired_split()
    assert len(c["probe"])==48 and len(c["heldout"])==48
    for v in ("probe","heldout"):
        assert all(c[v][i][1:]==c[v][i+1][1:] for i in range(0,len(c[v]),2))
