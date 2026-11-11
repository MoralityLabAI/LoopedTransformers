"""RMP-GG-01: registered positive-control machinery and independent oracles."""
from fractions import Fraction as Q
from itertools import product
import json
from pathlib import Path
import random

import pytest

torch=pytest.importorskip("torch")

from research_gym.scripts.goal_guarding_causal_pilot import (
    PRIORS,RATIOS, CUT, oracle, context, log_evaluation_odds, make_panel,
    RecurrentBeliefDecoder, patch_manifest, fit_belief_probe,
    measure_belief_probe, causal_patch_assay, run_pilot,
)


def alternative_oracle(prior,ratio,bits):
    # Independent Bernoulli enumeration using 3:1 likelihood ratios.
    ones=sum(bits); zeroes=len(bits)-ones
    return int(prior*ratio*Q(3)**ones > (1-prior)*Q(3)**zeroes)


def test_exact_rational_oracle_all_four_and_six_cue_contexts():
    for n in (0,1,2,4,6):
        for prior,ratio in product(PRIORS,RATIOS):
            for bits in product((0,1), repeat=n):
                assert oracle(prior,ratio,bits)==alternative_oracle(prior,ratio,bits)
    assert oracle(Q(1,2),Q(1),(0,1))==0  # Adverse tie.
    assert oracle(Q(1,2),Q(1),(1,1))==1
    with pytest.raises(ValueError,match="binary cue"):
        oracle(Q(1,2),Q(1),(2,))
    with pytest.raises(ValueError,match="preregistered"):
        oracle(Q(1,3),Q(1),(1,))


def test_panel_is_reproducible_and_prefix_labels_are_exact():
    a=make_panel(1024,4,67)
    b=make_panel(1024,4,67)
    assert torch.equal(a.bits,b.bits)
    assert torch.equal(a.contexts,b.contexts)
    assert torch.equal(a.actions,b.actions)
    for i in range(1024):
        row=tuple(map(int,a.bits[i].tolist()))
        assert a.actions[i].item()==oracle(a.priors[i],a.ratios[i],row)
        assert tuple(a.prefix_actions[i].tolist())==tuple(oracle(
            a.priors[i],a.ratios[i],row[:k]) for k in range(1,5))
        assert abs(a.prefix_log_odds[i].item()-log_evaluation_odds(
            a.priors[i],row[:CUT]))<1e-12
    assert not torch.equal(a.bits,make_panel(1024,4,68).bits)
    with pytest.raises(ValueError,match="positive"):
        make_panel(0,4,7)


def test_exact_counterfactual_manifest_is_frozen_and_discriminating():
    a=patch_manifest()
    b=patch_manifest()
    assert a==b and len(a)==153
    assert len(patch_manifest(cap=20))==20
    assert len(set((x.prior,x.ratio,x.receiver,x.donor) for x in a))==153
    for c in a:
        assert c.hybrid==c.donor[:CUT]+c.receiver[CUT:]
        assert c.target==oracle(c.prior,c.ratio,c.hybrid)
        assert c.target!=oracle(c.prior,c.ratio,c.receiver)
        assert c.target!=oracle(c.prior,c.ratio,c.donor)


def test_native_tied_block_receives_one_evidence_token_per_visit():
    torch.manual_seed(44)
    model=RecurrentBeliefDecoder()
    panel=make_panel(16,4,9)
    logits,states=model(panel.bits,panel.contexts)
    assert logits.shape==(16,4,2)
    assert states.shape==(16,4,32)
    assert sum(1 for module in model.modules()
               if module.__class__.__name__=="ExplicitDecoderBlock")==1
    assert not hasattr(model,"visit_embedding")
    with torch.no_grad():
        pre=model(panel.bits[:,:2],panel.contexts)[1][:,-1,:]
        future=model(panel.bits[:,2:],panel.contexts,initial_memory=pre)[0]
        assert torch.allclose(future,logits[:,2:,:],rtol=1e-5,atol=1e-5)
        altered=panel.bits.clone()
        altered[:,3]=1-altered[:,3]
        altered_logits,_=model(altered,panel.contexts)
        assert torch.allclose(altered_logits[:,:3],logits[:,:3])
    with pytest.raises(ValueError,match="context"):
        model.start(panel.contexts[:,:1])
    with pytest.raises(ValueError,match="binary evidence"):
        model(panel.bits[:,-1],panel.contexts)


def test_sham_and_full_state_patch_match_direct_hybrid_on_untrained_model():
    torch.manual_seed(44)
    model=RecurrentBeliefDecoder()
    cal=make_panel(256,4,10)
    probe=fit_belief_probe(model,cal)
    assert measure_belief_probe(model,make_panel(256,6,11),probe)>-1
    assay=causal_patch_assay(model,probe,patch_manifest())
    assert assay["oracle_manifest_size"]==153
    assert assay["eligible_after_model_accuracy_filter"]<=153
    if assay["assay_state"]=="measured":
        assert assay["full_state_identity_max_abs_error"]<1e-5
        assert len(assay["random_norm_matched"])==8


def test_smoke_receipt_checkpoints_and_controls(tmp_path:Path):
    report=run_pilot(tmp_path,steps=8,seeds=(11,),panel_size=128,train_size=512)
    assert report["protocol"]=="RMP-GG-01"
    assert report["scope"].startswith("supervised Bayesian")
    assert report["train_steps"]==8
    assert report["fixed_manifest_size"]==153
    saved=json.loads((tmp_path/"report.json").read_text())
    assert saved==report
    seed=report["seeds"][0]
    for arm,label in (("training","honest_supervised"),("shuffled_training","label_shuffle")):
        runs=seed[arm]
        assert runs["steps"]==8 and runs["arm"]==label
        for phase,key in (("initial","start_checkpoint_sha256"),("final","final_checkpoint_sha256")):
            path=tmp_path/"seed_11"/label/(phase+".pt")
            import hashlib
            assert hashlib.sha256(path.read_bytes()).hexdigest()==runs[key]
    assert seed["causal"]["oracle_manifest_size"]==153
    assert 0<=seed["initial_four_accuracy"]<=1
    assert 0<=seed["trained_six_accuracy"]<=1
    assert isinstance(seed["causal_support_gate"],bool)
