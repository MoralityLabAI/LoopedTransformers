"""PEF-07 standalone source split, teacher geometry, and frozen-model checks."""
import hashlib
import json
import torch
import pytest

from research.pef07.pef07 import (
    COUNTS, SEEDS, TASKS, EXTRA, D, source_split, encoded, as_tensor,
    target, Looper, model_digest, cohort_interventions, run, load_checkpoint,
    fit_ridge, apply_ridge
)


def test_all_4096_histories_are_context_disjoint_and_focal_closed():
    sources, manifest=source_split()
    assert manifest["sha256"] == hashlib.sha256(encoded({
        k:v for k,v in manifest.items() if k!="sha256"
    })).hexdigest()
    assert manifest["history_count_by_split"] == {
        "train":2560,"probe":768,"heldout":768
    }
    assert manifest["context_count_by_split"] == COUNTS
    assert len(manifest["contexts_in_sha256_order"]) == 256
    assert len(set(manifest["contexts_in_sha256_order"]))==256
    complete=set()
    for label,rows in sources.items():
        assert len(rows) == 16*COUNTS[label]
        for i in range(0,len(rows),16):
            group=rows[i:i+16]
            assert len({x[4:] for x in group})==1
            assert {x[:4] for x in group}=={
                tuple((mask>>j)&1 for j in range(4)) for mask in range(16)
            }
            for row in group:
                assert row not in complete
                complete.add(row)
    assert len(complete)==4096


def test_exact_teacher_balance_and_context_only_ceiling():
    catalog,_=source_split()
    x=as_tensor(catalog["heldout"])
    assert len(x)==768 and x.shape[1]==12
    for kind,ones,ceiling in (("coalition",288,480),("threshold",528,528)):
        y=target(x,kind)
        assert int(y.sum())==ones
        assert max(ones,768-ones)==ceiling
        baseline=torch.full_like(y,0 if kind=="coalition" else 1)
        assert int((baseline==y).sum())==ceiling
        for context_col in range(4,12):
            context_only=x[:,context_col]
            assert int((context_only==y).sum())<=ceiling
        for offset in range(0,768,16):
            assert int(y[offset:offset+16].sum())==(6 if kind=="coalition" else 11)
    with pytest.raises(ValueError):
        target(torch.zeros(4,8,dtype=torch.long),"coalition")
    with pytest.raises(ValueError):
        target(x,"missing")


def test_teacher_threshold_pair_only_can_have_additive_logits():
    cat,_=source_split()
    x=as_tensor(cat["heldout"])
    for task in TASKS:
        y=target(x,task).float()
        scalar=(x[:,:4].sum(dim=1).float()-1.5
                if task=="threshold" else y-0.5)
        hidden=torch.zeros((len(x),D),dtype=torch.float32)
        hidden[:,0]=scalar
        logits=torch.stack((torch.zeros_like(scalar),scalar),dim=1)
        evidence=cohort_interventions(x,hidden,logits,task)
        assert evidence["contexts"]==48
        assert len(evidence["pairs"])==6
        if task=="coalition":
            assert evidence["native_pair_only_events"]==96
            assert evidence["surrogate_mismatch_events"]==96
            assert evidence["surrogate_compatible_events"]==0
            assert evidence["intended_to_cross_ratio"]>1e5
        else:
            assert evidence["native_pair_only_events"]==288
            assert evidence["surrogate_mismatch_events"]==0
            assert evidence["surrogate_compatible_events"]==288
            assert evidence["intended_abs_mixed"]==0.0
            assert evidence["cross_abs_mixed"]==0.0
    wrong=x.clone()
    wrong[0,8]=1-wrong[0,8]
    with pytest.raises(AssertionError,match="closure"):
        cohort_interventions(wrong,hidden,logits,"coalition")


def test_random_tied_attention_uses_true_twelve_plus_two_natural_steps():
    torch.manual_seed(103)
    model=Looper().eval()
    catalog,_=source_split()
    bits=as_tensor(catalog["heldout"][:22])
    with torch.no_grad():
        h12=model.memory(bits)
        h13=model.step(h12,model.bit_embedding(torch.zeros(len(bits),dtype=torch.long)))
        h14=model.step(h13,model.bit_embedding(torch.ones(len(bits),dtype=torch.long)))
        automatic=model.memory(bits,EXTRA)
    assert torch.equal(automatic,h14)
    assert automatic.shape==(22,32)
    with pytest.raises(ValueError):
        model.memory(bits,(2,))


def test_frozen_checkpoint_precedes_scoring_and_debug_is_labeled(tmp_path):
    results=run(tmp_path,seeds=(103,),tasks=("coalition",),steps=2)
    assert not results["full_registered_run"]
    assert results["primary_decision"]=="nonregistered_debug_smoke"
    assert results["heldout_history_count"]==768
    assert results["all_models_sealed_before_any_probe_or_test"]
    ledger=json.loads((tmp_path/"sealed_training_ledger.json").read_text())
    assert ledger["all_six_trunks_sealed_before_scoring"]
    assert len(ledger["checkpoints"])==1
    rec=ledger["checkpoints"][0]
    assert rec["steps"]==2
    data=(tmp_path/rec["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest()==rec["checkpoint_sha256"]
    model=load_checkpoint(tmp_path,rec)
    assert model_digest(model)==rec["frozen_weights_sha256"]
    score=results["models"][0]
    assert score["trained_h12_native_head"]["total"]==768
    assert score["frozen_h14_downstream_ridge"]["total"]==768
    assert score["random_h14_downstream_ridge"]["total"]==768
    assert score["context_only_exact_ceiling"]==480
    assert score["focal_pair_effects"]["contexts"]==48
    assert score["all_original_weights_frozen_before_probe"]


def test_downstream_ridge_fitted_from_separate_source_only():
    sources,_=source_split()
    torch.manual_seed(10103)
    model=Looper().eval()
    original=model_digest(model)
    probe=as_tensor(sources["probe"][:48])
    heldout=as_tensor(sources["heldout"][:48])
    with torch.no_grad():
        a=model.memory(probe,EXTRA)
        b=model.memory(heldout,EXTRA)
        w=fit_ridge(a,target(probe,"coalition"))
        prediction=apply_ridge(b,w)
    assert tuple(prediction.shape)==(48,)
    assert model_digest(model)==original
