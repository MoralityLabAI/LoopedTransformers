"""GG-05P post-hoc: independently verify 8128 gold-deletion enumeration."""
from __future__ import annotations

from itertools import combinations

import pytest

torch=pytest.importorskip("torch")
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    EvaluationBeliefLooper,make_panel,_weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import query_splits
from research_gym.interp.goal_guarding_gold_influence_posthoc import (
    all_two_deletions,deletion_count,one_step_per_example,
    run_posthoc_gg05,
)


def test_all_double_deletions_match_independent_mask_oracle():
    # A negative gap comes entirely from the first two gold examples.
    examples=torch.tensor([
        [-10.,-10.,1.,1.,1.],
        [-7.,-6.,1.,2.,1.],
    ],dtype=torch.float64)
    full=all_two_deletions(examples)
    assert full["pairs_enumerated"]==10
    assert full["minimum_negative_gap_queries"]==0
    assert full["number_pairs_eliminating_all_negative_gaps"]==1
    assert full["pairs_attaining_minimum"]==[[0,1]]
    assert full["full_counts_by_pair"][(0,1)]==0
    for pair in combinations(range(5),2):
        assert full["full_counts_by_pair"][pair]==deletion_count(examples,pair)
    assert deletion_count(examples,())==2
    assert deletion_count(examples,(0,))==2
    assert deletion_count(examples,(0,1))==0


def test_two_sample_census_has_exact_8128_members():
    # Exact roster arithmetic does not depend on neural behavior.
    x=torch.ones(2,128,dtype=torch.float64)
    census=all_two_deletions(x)
    assert census["gold_examples"]==128
    assert census["source_queries"]==2
    assert census["pairs_enumerated"]==128*127//2==8128
    assert census["number_pairs_eliminating_all_negative_gaps"]==8128


def test_full_mask_and_census_validate_bounds_and_finiteness():
    with pytest.raises(ValueError,match="distinct"):
        deletion_count(torch.ones(2,5),(1,1))
    with pytest.raises(ValueError,match="distinct"):
        deletion_count(torch.ones(2,5),(10,))
    with pytest.raises(ValueError,match="every"):
        deletion_count(torch.ones(2,5),(0,1,2,3,4))
    with pytest.raises(ValueError,match="finite"):
        all_two_deletions(torch.tensor([[float("nan"),0.,1.]]))
    with pytest.raises(ValueError,match="finite"):
        all_two_deletions(torch.ones(2,2))


def test_native_two_action_sgd_returns_per_example_gold_contributions():
    torch.set_num_threads(1)
    torch.manual_seed(45)
    model=EvaluationBeliefLooper().eval()
    before=_weights_digest(model)
    gold=make_panel(8,6,68041)
    row=query_splits()["heldout"][0]
    effect=one_step_per_example(model,row,gold)
    assert effect.shape==(8,)
    assert torch.isfinite(effect).all()
    assert _weights_digest(model)==before
    # The per-example vector is a sufficient statistic for all deletions:
    # no SGD rerun is required for a different gold scoring subset.
    first=float(effect.mean())
    rest=float(effect[2:].mean())
    assert first==pytest.approx(float(effect.sum()/8))
    assert rest==pytest.approx(float((effect.sum()-effect[:2].sum())/6))
    assert isinstance(deletion_count(effect[None,:],(0,1)),int)


def test_runner_preserves_registered_source_scope(tmp_path):
    with pytest.raises(ValueError,match="original model seeds"):
        run_posthoc_gg05(source_dir=tmp_path,output=tmp_path/"test.json",
                        seeds=(11,))
