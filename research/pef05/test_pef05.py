"""PEF-05: sealed pair split, first-order projection, matched sham and fork audit."""
from __future__ import annotations
import hashlib
import json

import pytest
import torch

from research.pef04.pef04 import _canonical_bytes
from research.pef05.pef05 import (
    ARMS, SEEDS, WINDOW_END, WINDOW_START, pair_split, project, run
)


def test_fresh_pair_closed_split_and_manifest_hash():
    catalog,manifest=pair_split()
    assert manifest["history_counts"]=={"train":160,"probe":48,"heldout":48}
    assert manifest["pair_counts"]=={"train":80,"probe":24,"heldout":24}
    assert manifest["pair_closed"]
    assert manifest["primary_full_context_extension"]==[0,1]
    without={k:v for k,v in manifest.items() if k!="sha256"}
    assert hashlib.sha256(_canonical_bytes(without)).hexdigest()==manifest["sha256"]
    assert len(manifest["sorted_suffixes"])==128
    all_rows=set()
    for split,rows in catalog.items():
        assert len(rows)%2==0
        for a,b in zip(rows[0::2],rows[1::2]):
            assert (a[0],b[0])==(0,1)
            assert a[1:]==b[1:]
            assert a not in all_rows and b not in all_rows
            all_rows.update((a,b))
    assert len(all_rows)==256
    assert all(len(catalog[split])==2*n for split,n in
               (("train",80),("probe",24),("heldout",24)))


def test_orthogonal_projection_and_euclidean_norm_matched_sham():
    g=torch.tensor([1.,2.,3.,4.])
    m=torch.tensor([3.,-1.,2.,1.])
    a,ra=project(g,m,arm="protect",seed=157,step=126)
    b,rb=project(g,m,arm="sham",seed=157,step=126)
    assert ra["active"] and rb["active"]
    assert ra["memory_inner_before"]>0
    assert ra["memory_inner_after"]<=1e-5
    assert ra["matched_removed_gradient_norm_error"]<1e-5
    assert rb["matched_removed_gradient_norm_error"]<1e-5
    assert abs(float(a.norm())-float(b.norm()))<1e-5
    assert abs(ra["removed_l2"]-rb["removed_l2"])<1e-6
    assert not torch.allclose(a,b)


def test_no_projection_when_memory_gradients_indicate_increase():
    g=torch.tensor([-1.,0.,0.])
    m=torch.tensor([1.,0.,0.])
    for arm in ("protect","sham"):
        updated,row=project(g,m,arm=arm,seed=101,step=126)
        assert torch.equal(updated,g)
        assert row["active"] is False
    with pytest.raises(AssertionError,match="nonfinite"):
        project(g,torch.tensor([float("nan"),1.,0.]),
                arm="protect",seed=101,step=126)


def test_debug_branch_fork_and_probe_integrity_at_real_window(tmp_path):
    # Prefix at step149 and surgical update150; remaining two steps
    # vanilla. Mark explicitly NONREGISTERED, preserving the 125..150
    # intended full trial window in the real registered protocol.
    result=run(tmp_path,seeds=(157,),prefix_steps=149,total_steps=152)
    assert not result["registered_scoring_run"]
    assert result["registered_decision"]=="unregistered_debug"
    assert len(result["results"])==1
    assert result["results"][0]["heldout_history_count"]==48
    assert result["results"][0]["paired_counterfactual_count"]==24
    ledger=json.loads((tmp_path/"seed157.sealed_ledger.json").read_text())
    assert ledger["all_three_arms_frozen_before_downstream_probe"]
    assert len({ledger["arms"][arm]["batch_stream_sha256"] for arm in ARMS})==1
    assert len({ledger["arms"][arm]["common_weight_sha256"] for arm in ARMS})==1
    assert len({ledger["arms"][arm]["common_optimizer_sha256"] for arm in ARMS})==1
    assert len(ledger["arms"]["protect"]["surgery"])==1
    assert len(ledger["arms"]["sham"]["surgery"])==1
    assert len(ledger["arms"]["baseline"]["surgery"])==0
    for arm in ARMS:
        assert ledger["arms"][arm]["intermediate"]["step"]==150
        assert ledger["arms"][arm]["final"]["step"]==152
        assert result["results"][0]["arms"][arm]["h10_01"]["total_pairs"]==24
        assert result["results"][0]["arms"][arm]["h10_01"]["total_histories"]==48
    for arm in ("protect","sham"):
        row=ledger["arms"][arm]["surgery"][0]
        assert row["step"]==150
        assert row["matched_removed_gradient_norm_error"]<1e-5
    assert result["results"][0]["source_only_correct_out_of_48"]==24
    assert result["results"][0]["source_only_joint_pairs_out_of_24"]==0
