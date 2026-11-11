"""GG-01 exact oracle, tied decoder, counterfactual and negative controls."""
from fractions import Fraction as Q
from itertools import product
import math

import pytest

torch=pytest.importorskip("torch")

from research_gym.interp.goal_guarding_causal import (
    PRIORS, RATIOS, PATCH_VISIT, bayes_masses, exact_action,
    exact_posterior_odds, make_panel, EvaluationBeliefLooper,
    train_cell, fit_belief_probe, counterfactual_pairs,
    causal_intervention, run_pilot,
)


def test_exact_bayes_oracle_and_posterior_odds_full_enumeration():
    for p, ratio in product(PRIORS,RATIOS):
        for k in range(1,7):
            for bits in product((0,1),repeat=k):
                e,d=bayes_masses(p,ratio,bits)
                assert e>=0 and d>=0
                assert exact_action(p,ratio,bits)==int(e>d)
                posterior=exact_posterior_odds(p,bits)
                expected=p/(1-p)*Q(3)**(2*sum(bits)-k)
                assert posterior==expected
    assert exact_action(Q(1,2),Q(1),(1,0))==0
    assert exact_action(Q(1,2),Q(1),(1,1))==1
    with pytest.raises(ValueError,match="binary"):
        exact_action(Q(1,2),Q(1),(2,))
    with pytest.raises(ValueError,match="interior"):
        exact_action(Q(1),Q(1),(0,))


def test_deterministic_panel_and_heldout_depth():
    a=make_panel(128,4,101)
    b=make_panel(128,4,101)
    null=make_panel(128,4,101,shuffled_labels=True)
    long=make_panel(128,6,101)
    assert torch.equal(a.observations,b.observations)
    assert torch.equal(a.labels,b.labels)
    assert torch.equal(a.observations,null.observations)
    assert sorted(a.labels.tolist())==sorted(null.labels.tolist())
    assert not torch.equal(a.labels,null.labels)
    assert long.observations.shape==(128,6)
    assert all(
        int(a.labels[i])==exact_action(*a.parameters[i],tuple(a.observations[i].tolist()))
        for i in range(len(a))
    )
    assert all(
        torch.equal(a.prefix_labels[i],
          torch.tensor([exact_action(*a.parameters[i],tuple(a.observations[i,:k].tolist()))
                        for k in range(1,5)]))
        for i in range(len(a))
    )


def test_native_tied_recurrent_state_patch_is_exact_hybrid_identity():
    torch.manual_seed(4)
    torch.set_num_threads(1)
    model=EvaluationBeliefLooper(width=16,heads=4).eval()
    receiver=torch.tensor([[1,0,1,0],[0,0,1,1],[1,1,0,0]],dtype=torch.long)
    donor=torch.tensor([[0,1,0,0],[1,1,1,0],[0,0,1,1]],dtype=torch.long)
    ctx=torch.zeros((3,2))
    hybrid=torch.cat((donor[:,:2],receiver[:,2:]),dim=1)
    with torch.no_grad():
        donor_states=model(donor,ctx)[1][:,1,:]
        recv_logits,recv_states=model(receiver,ctx)
        direct=model(hybrid,ctx)[0]
        patched=model(receiver,ctx,patch_visit=2,patch_state=donor_states)[0]
        sham=model(receiver,ctx,patch_visit=2,patch_state=recv_states[:,1,:])[0]
    assert torch.allclose(patched[:,-1,:],direct[:,-1,:],rtol=2e-5,atol=2e-5)
    assert torch.allclose(sham,recv_logits,rtol=2e-5,atol=2e-5)
    assert model.block is model.block
    with pytest.raises(ValueError,match="patch_state"):
        model(receiver,ctx,patch_visit=2)
    with pytest.raises(ValueError,match="patch_visit"):
        model(receiver,ctx,patch_state=donor_states)


def test_manifest_uses_independent_oracle_and_excludes_donor_copy():
    manifest=counterfactual_pairs(64,visits=4,cutoff=PATCH_VISIT)
    assert manifest.receiver.shape==(64,4)
    assert manifest.donor.shape==(64,4)
    assert manifest.attempted>=64
    assert torch.equal(manifest.hybrid[:,:2],manifest.donor[:,:2])
    assert torch.equal(manifest.hybrid[:,2:],manifest.receiver[:,2:])
    # Recover context from finite pre-registered options.
    contexts={(round(math.log(float(p/(1-p))),6),
              round(math.log(float(r)),6)):(p,r)
              for p in PRIORS for r in RATIOS}
    for i in range(64):
        key=tuple(round(float(x),6) for x in manifest.context[i])
        # float32 context round-off can perturb the final decimal.
        p,r=min(((p,r) for p in PRIORS for r in RATIOS),
                key=lambda item:sum(abs(float(manifest.context[i,j])-v)
                 for j,v in enumerate((math.log(float(item[0]/(1-item[0]))),
                                        math.log(float(item[1]))))))
        recv=tuple(manifest.receiver[i].tolist())
        donor=tuple(manifest.donor[i].tolist())
        hybrid=tuple(manifest.hybrid[i].tolist())
        assert exact_action(p,r,hybrid)==int(manifest.labels[i])
        assert exact_action(p,r,hybrid)!=exact_action(p,r,recv)
        assert exact_action(p,r,hybrid)!=exact_action(p,r,donor)


def test_small_cpu_training_and_direction_assay(tmp_path):
    model,record=train_cell(11,steps=120,batch=64,save_dir=tmp_path)
    assert record["updates"]==120
    assert record["initial_state_digest"]!=record["final_state_digest"]
    assert (tmp_path/record["artifact_name"]).exists()
    assert 0<=record["heldout_4_accuracy"]<=1
    assert 0<=record["heldout_6_accuracy"]<=1
    probe=fit_belief_probe(model)
    assert probe.direction.shape==(32,)
    assert torch.isfinite(probe.direction).all()
    assert abs(float(probe.direction.norm())-1)<1e-5
    assert all(math.isfinite(x) for x in (
        probe.calibration_r2,probe.validation_r2,probe.depth6_r2))
    report=causal_intervention(model,probe,pairs=64,random_controls=3)
    assert report["attempted"]>=64
    assert report["manifest"]==64
    assert 0<=report["eligible"]<=64
    if report["eligible"]:
        assert report["full_equal_to_exact_hybrid"]
        assert report["sham_equal_to_receiver"]
        assert report["full_patch"]["hit"]==1
        assert report["receiver"]["hit"]==0
        assert len(report["random_controls"])==3
        assert report["donor_copy_label_disagrees_by_oracle"]


def test_zero_step_and_shuffled_control_smoke(tmp_path):
    model,record=train_cell(17,steps=0,save_dir=tmp_path)
    assert record["initial_state_digest"]==record["final_state_digest"]
    assert record["loss_checkpoints"]==[]
    null_model,null=train_cell(9011,steps=3,batch=32,shuffled_labels=True)
    assert null["shuffled_labels"] is True
    report=run_pilot(steps=3,seeds=(11,),include_shuffled_control=False)
    assert report["kind"]=="supervised neural causal positive control"
    assert report["trained_model_causal_claim"] is False
    assert len(report["results"])==1
    assert isinstance(report["all_seeds_eligible"],bool)
