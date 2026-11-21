"""NEC-03N fixed-native 900->1800 source-only learning contracts."""
from __future__ import annotations

import pytest
import torch

from mechinterp.neuron_communities.nec03n_native1800 import (
    SCHEMA,INDICES,OLD_STEPS,NEW_STEPS,BATCH,LEARNING_RATE,
    COMPETENCE_FLOOR,SOURCE_HISTORY_COUNT,source_confusion,run,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from research.pef07.pef07 import Looper,model_digest


def test_1800_source_only_training_schedule_and_cohort_fixed():
    assert INDICES==(0,1,2,3)
    assert OLD_STEPS==900
    assert NEW_STEPS==1800
    assert BATCH==128
    assert LEARNING_RATE==.003
    assert COMPETENCE_FLOOR==.9
    assert SOURCE_HISTORY_COUNT==2560
    assert len(registration()["new_five_input_development_world_specs"])==96
    assert len(registration()["new_five_input_embargoed_world_commitments"])==32
    assert len(old_registry()["confirmation_commitments"])==64


def test_full_source_confusion_uses_all_2560_input_histories_and_class_truth():
    torch.set_num_threads(1)
    torch.manual_seed(301)
    model=Looper().eval()
    world=worlds()[0]
    catalog,_=context_catalog(world)
    original=model_digest(model)
    for arm in ("composed","additive"):
        data=source_confusion(model,catalog["train"],world,arm=arm)
        assert data["total"]==2560
        assert len(data["confusion_truth_rows_pred_columns"])==2
        assert sum(sum(row) for row in data["confusion_truth_rows_pred_columns"])==2560
        assert sum(data["truth_class_counts"])==2560
        assert data["source_training_accuracy_only"]
        assert data["correct"] in range(2561)
        assert data["accuracy"]==data["correct"]/2560
        assert data["class_recall"][0] is not None
        assert data["class_recall"][1] is not None
    assert model_digest(model)==original
    with pytest.raises(ValueError,match="2,560"):
        source_confusion(model,catalog["train"][:32],world,arm="composed")


def test_calibration_refuses_untouched_dev_or_embargoed_world_before_old_model_read(tmp_path):
    for idx in (4,5,95,96,127):
        with pytest.raises(PermissionError):
            run(idx,tmp_path/"never-registered-original",tmp_path/"never-output")


def test_true_five_input_source_only_model_checkpoint_requires_96dev_32embargo():
    for i in INDICES:
        w=worlds()[i]
        c,_=context_catalog(w)
        assert len(c["train"])==2560
        assert len(c["development"])==768
        assert len(c["reserved"])==768
        assert w["cohort"]=="development"
    for i in (96,127):
        assert worlds()[i]["cohort"]=="embargoed"
        with pytest.raises(PermissionError):
            context_catalog(worlds()[i])
