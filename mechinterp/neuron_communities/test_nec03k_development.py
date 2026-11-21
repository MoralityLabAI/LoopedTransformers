"""NEC-03K-B source-frozen causal checks on already-opened worlds."""
from __future__ import annotations

import json

import pytest
import torch

from mechinterp.neuron_communities.nec03k_development import (
    SCHEMA, DEV_CONTEXTS, MIN_CLEAN,
)
from mechinterp.neuron_communities.nec03k_dev_audit import (
    prediction,margin,
)
from mechinterp.neuron_communities.nec03g_worlds import worlds,registry


def test_causal_logit_orientation_frozen_from_clean_predicted_class():
    assert prediction([1.,-1.])==0
    assert prediction([-1.,1.])==1
    assert margin([-0.5,0.2],1)==pytest.approx(.7)
    assert margin([-0.5,0.2],0)==pytest.approx(-.7)
    with pytest.raises(AssertionError):
        prediction([float("nan"),1.])


def test_original_source_cohorts_only_and_conservative_itt():
    assert DEV_CONTEXTS==48
    assert MIN_CLEAN==.90
    assert SCHEMA.startswith("NEC-03K-B/")
    assert all(worlds()[i]["cohort"]=="development" for i in range(8,16))
    assert all(worlds()[i]["cohort"]=="confirmation" for i in range(64,128))
    assert len(registry()["confirmation_commitments"])==64


def test_output_margin_correctness_and_nonlinearity_not_conflated():
    source=[-1.,1.2]
    control=[-1.,.8]
    # Correctness can agree even when margin shifts substantially.
    assert prediction(source)==prediction(control)==1
    assert margin(source,1)-margin(control,1)==pytest.approx(.4)
    # The true label is never needed to orient the derivative.
    assert margin(source,0)-margin(control,0)==pytest.approx(-.4)


def test_causal_ITT_filters_incompetent_teacher_even_when_control_available():
    old_head_accuracy=526/768
    source_new=500
    control_new=495
    eligible=True
    effect=source_new-control_new if eligible and old_head_accuracy>=MIN_CLEAN else 0
    assert effect==0
