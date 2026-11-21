"""NEC-03N-B tests: checkpoint reuse, source-cohort and offset seed budget."""
from __future__ import annotations

import pytest
import torch

from mechinterp.neuron_communities.nec03nb_checkpoint_reset import (
    OLD_STEPS,ADDITIONAL_STEPS,TOTAL_STEPS,SOURCE_STATES,
    ROOTS,CLEAN_SOURCE_FLOOR,batch_digest,source_confusion,run,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import worlds,context_catalog
from research.pef07.pef07 import Looper


def test_registered_only_four_original_five_input_worlds_and_900_plus_900_budget():
    assert ROOTS==(0,1,2,3)
    assert (OLD_STEPS,ADDITIONAL_STEPS,TOTAL_STEPS)==(900,900,1800)
    assert SOURCE_STATES==2560
    assert CLEAN_SOURCE_FLOOR==.90
    assert all(worlds()[i]["cohort"]=="development" for i in ROOTS)
    assert all(worlds()[i]["cohort"]=="embargoed" for i in range(96,128))


def test_original_seeded_batch_generator_continuation_after_900_consumed_batches():
    seed=5007
    rng=torch.Generator().manual_seed(seed*7919+59)
    old=[torch.randint(SOURCE_STATES,(128,),generator=rng) for _ in range(900)]
    next_one=torch.randint(SOURCE_STATES,(128,),generator=rng)
    # Resetting the optimizer must preserve the *source sample index stream*
    # by advancing the independent PyTorch generator exactly 900 batches.
    rng2=torch.Generator().manual_seed(seed*7919+59)
    for _ in range(900):
        torch.randint(SOURCE_STATES,(128,),generator=rng2)
    next_two=torch.randint(SOURCE_STATES,(128,),generator=rng2)
    assert batch_digest(next_one)==batch_digest(next_two)
    assert batch_digest(old[0])!=batch_digest(next_one)


def test_native_source_scoring_is_five_input_80_by_32_in_sample_only():
    torch.set_num_threads(1)
    torch.manual_seed(113)
    model=Looper().eval()
    world=worlds()[0]
    catalog,_=context_catalog(world)
    composed=source_confusion(model,catalog["train"],world,arm="composed")
    additive=source_confusion(model,catalog["train"],world,arm="additive")
    assert composed["total"]==additive["total"]==2560
    assert len(composed["source_prediction_labels"])==2560
    assert sum(map(sum,composed["confusion_truth_rows_pred_columns"]))==2560
    assert composed["training_source_only"]
    assert 0<=composed["accuracy"]<=1
    with pytest.raises(ValueError):
        source_confusion(model,catalog["development"],world,arm="composed")


def test_forbidden_world_enrollment_rejected_before_loading_checkpoint(tmp_path):
    from mechinterp.neuron_communities.nec03nb_checkpoint_reset import verify_lock
    lock=verify_lock()
    assert lock["exact_continuous_AdamW_1800_claim"] is False
    for idx in (4,95,96,127):
        with pytest.raises(PermissionError,match="previously trained"):
            run(idx,tmp_path/"no_original",tmp_path/"new")


def test_frozen_original_NEC03M_source_lock_does_not_migrate_to_new_condition():
    from mechinterp.neuron_communities.nec03nb_checkpoint_reset import verify_lock
    lock=verify_lock()
    assert lock["original_NEC03M_run_id"]==44894980345
    assert lock["failed_exact_continuation_run_id"]==44895917012
    assert lock["original_checkpoint_training_steps"]==900
    assert lock["new_steps_with_reset_AdamW"]==900
    assert lock["total_model_updates"]==1800
    assert lock["new_sae_models_trained"]==0
    assert lock["development_eval_contexts_scored"]==0
    assert lock["original_four_input_confirmation_worlds_scored"]==0
