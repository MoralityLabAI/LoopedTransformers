"""PEF-04B smoke: same-init hash and natural paired trajectory."""
import torch
from research.pef03.pef03 import BitMemoryLooper,hash_weights,tensor
from research.pef04.pef04 import paired_split,run
from research.pef04.pef04_matched_init import natural_trajectory,arm_score


def test_initial_weights_are_reconstructed_from_exact_source_seed(tmp_path):
    result=run(tmp_path,seeds=(101,),steps=2)
    import json
    ledger=json.loads((tmp_path/"sealed_training_ledger.json").read_text())
    torch.manual_seed(101)
    initialized=BitMemoryLooper().eval()
    assert hash_weights(initialized)==ledger["training"][0]["initial_weights_sha256"]
    cat,_=paired_split()
    with torch.no_grad():
        rows=natural_trajectory(initialized,tensor(cat["heldout"]))
        arm=arm_score(initialized,tensor(cat["probe"]),tensor(cat["heldout"]),(0,0))
    assert len(rows)==10
    assert abs(rows[-1]["mean_oldest_bit_pair_l2"]-arm["mean_natural_pair_hidden_l2"])<1e-5
    assert arm["total_pairs"]==24
    assert 0<=arm["both_members_correct_pairs"]<=24
    assert result["registered_scoring_run"] is False


def test_trajectory_source_pair_closure():
    cat,_=paired_split()
    assert len(cat["heldout"])==48
    assert all(cat["heldout"][i][1:]==cat["heldout"][i+1][1:]
               for i in range(0,48,2))
