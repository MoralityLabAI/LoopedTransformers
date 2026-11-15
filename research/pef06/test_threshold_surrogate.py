"""PEF-06T: explicit additive-threshold and nonlinear-logit negative controls.

All tests use complete 256-word cubes, preserving original task bits 6/7.
"""
import torch
import pytest
from research.pef06.threshold_surrogate import threshold_surrogate
from research.pef06.frozen_coalitions import source_histories, latch


def _example(*, nonlinear: bool):
    x=source_histories()
    if nonlinear:
        response=3*(x[:,0].float()*x[:,1].float())-1
        h=x[:,0].float()*x[:,1].float()
    else:
        response=x[:,0].float()+x[:,1].float()-1.5
        h=x[:,0].float()+x[:,1].float()
    hidden=torch.zeros((256,16),dtype=torch.float32)
    hidden[:,0]=h
    logits=torch.full((256,4),-100.,dtype=torch.float32)
    logits[:,0]=0
    logits[:,1]=response
    return logits,hidden


def test_additive_scores_generate_pair_only_class_changes_without_interaction():
    scores,features=_example(nonlinear=False)
    r=threshold_surrogate(scores,features)
    row=next(z for z in r['per_pair'] if z['history_bit_pair']==[0,1])
    assert row['pair_only_actual_task_class_changes']==64
    assert row['pair_only_exact_class_reproduced_by_additive_logit_surrogate']==64
    assert row['pair_only_class_differs_from_additive_logit_surrogate']==0
    assert row['mean_four_corner_logit_l2'] == pytest.approx(0.)
    assert row['mean_four_corner_hidden_l2'] == pytest.approx(0.)
    assert r['pair_only_events_not_reproduced_by_additive_surrogate']==0
    assert r['pair_only_events_reproduced_by_additive_surrogate']==64


def test_nonlinear_and_gate_requires_nonadditive_logit_second_difference():
    scores,features=_example(nonlinear=True)
    r=threshold_surrogate(scores,features)
    row=next(z for z in r['per_pair'] if z['history_bit_pair']==[0,1])
    assert row['pair_only_actual_task_class_changes']==64
    assert row['pair_only_class_differs_from_additive_logit_surrogate']==64
    assert row['pair_only_exact_class_reproduced_by_additive_logit_surrogate']==0
    assert row['mean_four_corner_logit_l2'] > 0
    assert row['mean_four_corner_hidden_l2'] > 0
    assert r['pair_only_events_not_reproduced_by_additive_surrogate']==64


def test_xor_and_latch_controls_expose_independent_two_bit_events():
    inputs=source_histories()
    word=torch.arange(256)
    outcome=torch.tensor([latch(int(i)) for i in word],dtype=torch.float32)
    state=torch.zeros((256,16))
    state[:,0]=outcome
    logits=torch.full((256,4),-100.)
    logits[:,0]=0
    logits[:,1]=2*outcome-1
    r=threshold_surrogate(logits,state)
    for pair in ([0,1],[2,3],[4,5]):
        row=next(z for z in r['per_pair'] if z['history_bit_pair']==pair)
        assert row['pair_only_actual_task_class_changes']>0
        assert row['mean_four_corner_logit_l2']>0


def test_target_conditional_and_surrogate_counts_partition_every_pair():
    torch.manual_seed(140119)
    logits=torch.randn(256,4)
    hidden=torch.randn(256,16)
    r=threshold_surrogate(logits,hidden)
    assert r['source_histories']==256
    assert r['old_bit_intervention_pairs']==15
    assert r['count_denominator_all_source_pair_cases']==3840
    assert len(r['per_pair'])==15
    for row in r['per_pair']:
        assert (row['pair_only_from_correct_base_task_prediction']+
                row['pair_only_from_incorrect_base_task_prediction']==
                row['pair_only_actual_task_class_changes'])
        assert (row['pair_only_exact_class_reproduced_by_additive_logit_surrogate']+
                row['pair_only_class_differs_from_additive_logit_surrogate']==
                row['pair_only_actual_task_class_changes'])
        assert sum(row['pair_only_cases_by_supervised_target_class'].values())==row[
            'pair_only_actual_task_class_changes']


def test_invalid_shapes_and_nonfinite_values_fail_closed():
    with pytest.raises(ValueError):
        threshold_surrogate(torch.zeros((255,4)),torch.zeros((255,16)))
    logits=torch.zeros((256,4))
    logits[0,0]=float('nan')
    with pytest.raises(AssertionError,match="nonfinite"):
        threshold_surrogate(logits,torch.zeros((256,16)))
