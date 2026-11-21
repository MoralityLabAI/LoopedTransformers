"""NEC-03I pure-stdlib tests of tamper-resistant causal source diagnostics."""
import tempfile
from pathlib import Path

import pytest

from mechinterp.neuron_communities.nec03i_trace_audit import (
    aggregate, clean_hits, verify_feature_policy, verify_physical_dose,
)


def test_correctness_computed_from_raw_logits_and_mask_teacher():
    y=[i%2 for i in range(16)]
    good=[[1.,-1.] if truth==0 else [-1.,1.] for truth in y]
    hits=clean_hits(good,y)
    assert hits==[1]*16
    changed=[row[:] for row in good]
    changed[4]=changed[4][::-1]
    assert sum(clean_hits(changed,y))==15
    with pytest.raises(AssertionError,match="incomplete"):
        clean_hits(good[:15],y)
    with pytest.raises(AssertionError,match="nonfinite"):
        clean_hits([[float("nan"),1.]]+good[1:],y)


def _receipt(a=2.,b=1.,t=1.,gleft=.5,gright=1.):
    return {
        "full_left_dose":[a]*16,
        "full_right_dose":[b]*16,
        "shared_achievable_dose":[t]*16,
        "realized_left_dose":[t]*16,
        "realized_right_dose":[t]*16,
        "left_gain":[gleft]*16,
        "right_gain":[gright]*16,
        "retained_fraction_of_full_left_dose":[t/a]*16,
        "relative_dose_mismatch":[0.]*16,
        "matched_within_1pct":[True]*16,
        "shared_nonzero":[True]*16,
    }


def test_physical_dose_verifier_checks_no_amplification_and_original_strength():
    r=_receipt()
    out=verify_physical_dose(r)
    assert len(out)==16 and all(x==(1.,.5,True) for x in out)
    r2=_receipt(gleft=2.)
    with pytest.raises(AssertionError,match="amplification"):
        verify_physical_dose(r2)
    r3=_receipt(a=5.)
    with pytest.raises(AssertionError,match="dose"):
        verify_physical_dose(r3)
    r4=_receipt()
    r4["retained_fraction_of_full_left_dose"]=[1.]*16
    with pytest.raises(AssertionError,match="fraction"):
        verify_physical_dose(r4)


def test_ranked_active_feature_selection_does_not_see_teacher_labels():
    rank=[list(range(128)) for _ in range(12)]
    policy={"0":{
        "all_active_features":[list(range(16)) for _ in range(16)],
        "top_source_ranked_active":[[0,1] for _ in range(16)],
        "bottom_source_ranked_active":[[15,14] for _ in range(16)],
        "selected_features_per_arm_per_visit":2,
    }}
    score=verify_feature_policy(policy,rank)
    assert score["routed_features"]==32
    assert score["unique_selected_features_per_visit"]["0"]==2
    bad={"0":dict(policy["0"])}
    bad["0"]["top_source_ranked_active"]=[[14,15] for _ in range(16)]
    with pytest.raises(AssertionError,match="selected incorrectly"):
        verify_feature_policy(bad,rank)


def test_guard_fails_closed_without_original_eight_world_artifacts(tmp_path):
    lock=Path(__file__).with_name("NEC03H_V2_LOCK.json")
    with pytest.raises(AssertionError,match="exactly the eight"):
        aggregate(tmp_path,lock_path=lock,output=tmp_path/"out")
