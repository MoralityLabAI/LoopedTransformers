"""NEC-03Q locked development causal pilot, including conservative missing sham."""
from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch

from mechinterp.neuron_communities.nec03p_causal_manifest import development_pilot_manifest
from mechinterp.neuron_communities.nec03q_causal_run import (
    ROOTS,ALL_ROOTS,DEV_CASES,MIN_MEANINGFUL_EFFECT,
    verify_lock,paired_dev_group,run,
)
from mechinterp.neuron_communities.nec03q_aggregate import exact_two_sided_root_sign_flip
from mechinterp.neuron_communities.nec03l_five_input_worlds import worlds,context_catalog
from mechinterp.neuron_communities.topk_sae import TopKSAE
from research.pef07.pef07 import Looper


def test_sealed_parent_and_exact_development_cohort_lock():
    lock,lock_sha=verify_lock()
    assert len(lock_sha)==64
    assert ROOTS==(0,1,2,4,6,7)
    assert ALL_ROOTS==tuple(range(8))
    assert DEV_CASES==768
    assert MIN_MEANINGFUL_EFFECT==.02
    assert lock["development_root_indices"]==list(ROOTS)
    assert lock["unavailable_root_indices_assigned_zero"]==[3,5]
    assert lock["eight_root_ITT_denominator"]==list(range(8))
    assert lock["familywise_primary_comparisons"]==1
    assert lock["reserved_contexts_scored"]==0
    assert lock["embargoed_worlds_scored"]==0
    assert lock["original_four_input_confirmation_worlds_scored"]==0
    assert development_pilot_manifest()["pilot_readout_status"]=="SEALED_UNOPENED"


def test_root3_and_root5_are_blocked_prior_to_loading_any_model(tmp_path):
    for index in (3,5,8,31,96,127,-1):
        with pytest.raises(PermissionError):
            run(index,tmp_path/"source",tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_missing_16_active_codes_gives_both_donors_identical_hubs_only():
    torch.set_num_threads(1)
    torch.manual_seed(44)
    model=Looper().eval()
    sae=TopKSAE().eval()
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.zero_()
    w=worlds()[0]
    catalog,_=context_catalog(w)
    loaded={
        "model":model,"sae":sae,"mean":torch.zeros(64),"scale":torch.ones(64),
        "ranks":[list(range(128)) for _ in range(12)],
        "chosen":{0:[3]},"hubs":{},
    }
    receipt=paired_dev_group(loaded,catalog["train"][:32])
    assert sum(receipt["valid_input_control"])==0
    assert receipt["source_logits"]==receipt["sham_logits"]
    assert receipt["site_geometry"]["0"]["candidate_count"]==[0]*32
    assert receipt["true_labels_used_in_donor_selection"] is False


def test_sign_flip_is_exact_and_pvalue_is_exploratory():
    assert exact_two_sided_root_sign_flip([0.,0.,0.,0.,0.,0.,0.,0.])==(1.0,0)
    p,n=exact_two_sided_root_sign_flip([.1,.1,.1,.1,0.,0.,0.,0.])
    assert n==4 and math.isclose(p,.125)
