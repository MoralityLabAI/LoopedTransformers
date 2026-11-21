"""NEC-03E: read-only, attenuation-only pairwise native/SAE dose experiment.

Follow-up to NEC-03D's nine-of-nine failed 8x gain-cap comparisons.
At each recurrent visit and input, the two donors in a registered pair
are each attenuated to min(their original local MLP-down output norms).
No donor is amplified. Exact paired outcomes and retained source dose
are archived; the PEF-07 heldout contexts were previously opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities.core import decode, interval
from mechinterp.neuron_communities.matched_dose import (
    by_visit, masked, physical_dose, graft_forward,
)
from mechinterp.neuron_communities.nec03_development import (
    _load_prepared, _feature_indices, frozen_model, serialized,
)
from mechinterp.neuron_communities.source_sae import (
    L, WIDTH, clean_activations, physical_feature_delta,
)
from research.pef07.pef07 import as_tensor, model_digest, target

SCHEMA="NEC-03E/attenuation-only-two-way-matched-physical-dose/v1"
PAIRS=(("native","sham"),("native","sae"),("sae","sae_sham"))
EPS=1e-10
DOSE_MATCH_TOLERANCE=0.01
MIN_ACTIVE_FRAC=0.80
MIN_MEDIAN_RETAINED=0.25


def attenuate_pair(left, right, down_weight):
    """Identical direct physical dose using gain in [0,1] for both arms."""
    if left.shape!=right.shape or left.ndim!=2 or left.shape[1]!=WIDTH:
        raise ValueError("wrong donor pair shape")
    norm_l=physical_dose(left,down_weight)
    norm_r=physical_dose(right,down_weight)
    target=torch.minimum(norm_l,norm_r)
    gain_l=torch.where(norm_l>EPS,target/norm_l.clamp_min(EPS),
                       torch.zeros_like(norm_l))
    gain_r=torch.where(norm_r>EPS,target/norm_r.clamp_min(EPS),
                       torch.zeros_like(norm_r))
    if (gain_l>1+1e-6).any() or (gain_r>1+1e-6).any():
        raise AssertionError("an attenuated pair amplified its donor")
    gain_l=gain_l.clamp(0,1)
    gain_r=gain_r.clamp(0,1)
    new_l=left*gain_l[:,None]
    new_r=right*gain_r[:,None]
    physical_l=physical_dose(new_l,down_weight)
    physical_r=physical_dose(new_r,down_weight)
    mismatch=(physical_l-physical_r).abs()/target.clamp_min(1e-6)
    relative_retained=torch.where(norm_l>1e-6,target/norm_l.clamp_min(EPS),
                                  torch.zeros_like(norm_l))
    return new_l,new_r,{
        "full_left_dose":norm_l.tolist(),
        "full_right_dose":norm_r.tolist(),
        "shared_achievable_dose":target.tolist(),
        "realized_left_dose":physical_l.tolist(),
        "realized_right_dose":physical_r.tolist(),
        "left_gain":gain_l.tolist(),
        "right_gain":gain_r.tolist(),
        "relative_dose_mismatch":mismatch.tolist(),
        "matched_within_1pct":(mismatch<=DOSE_MATCH_TOLERANCE).tolist(),
        "shared_nonzero":(target>1e-6).tolist(),
        "retained_fraction_of_full_left_dose":relative_retained.tolist(),
        "method":"min(left,right) in frozen mlp_down output L2; attenuation only",
    }


@torch.no_grad()
def candidate_donors(model, clean_acts, candidate, ranking, hubs):
    selected=by_visit(candidate["nodes"])
    control=by_visit(ranking["hub_disjoint_native_sham"])
    hubs_by=by_visit(hubs)
    ftop=_feature_indices(ranking,"sae_feature_indices_by_visit")
    falt=_feature_indices(ranking,"sae_random_features_by_visit")
    if not set(selected)==set(control)==set(ftop)==set(falt):
        raise AssertionError("same recurrent visit budget required")
    for visit in selected:
        if (len(selected[visit])!=len(control[visit])
            or len(selected[visit])!=len(ftop[visit])
            or len(selected[visit])!=len(falt[visit])):
            raise AssertionError("feature/native cardinality per visit differs")
        if selected[visit]&hubs_by.get(visit,set()) or control[visit]&hubs_by.get(visit,set()):
            raise AssertionError("hub overlap despite source repair")
    return selected,control,hubs_by,ftop,falt


@torch.no_grad()
def make_pair(model, clean_acts, candidate, ranking,
              hubs, sae, mean, scale, pair):
    if pair not in PAIRS:
        raise ValueError("unknown fixed pair family")
    selected,control,hubs_by,ftop,falt=candidate_donors(
        model,clean_acts,candidate,ranking,hubs)
    weight=model.block.mlp_down.weight.detach()
    donors,doses,masks={},{},{}
    for visit in sorted(selected):
        a=clean_acts[:,visit,:]
        inputs={
            "native":masked(a,selected[visit]),
            "sham":masked(a,control[visit]),
            "sae":physical_feature_delta(sae,mean,scale,a,ftop[visit]),
            "sae_sham":physical_feature_delta(sae,mean,scale,a,falt[visit]),
        }
        left,right,dose=attenuate_pair(
            inputs[pair[0]],inputs[pair[1]],weight)
        donors[visit]={
            "hubs":torch.zeros_like(left),
            "native":left,"sham":right,
        }
        masks[visit]=masked(torch.ones_like(a),hubs_by.get(visit,set()))[0]
        doses[visit]=dose
    return donors,doses,masks


def pair_dose_stats(per_context):
    """Site observations diagnose dose feasibility, not sample size for p-values."""
    t=[]
    matched=[]
    active=[]
    ratio=[]
    for row in per_context:
        for dose in row.values():
            t.extend(dose["shared_achievable_dose"])
            matched.extend(dose["matched_within_1pct"])
            active.extend(dose["shared_nonzero"])
            ratio.extend(dose["retained_fraction_of_full_left_dose"])
    if not t:
        raise AssertionError("no dose data")
    denom=len(t)
    s=sorted(ratio)
    median=(s[(len(s)-1)//2]+s[len(s)//2])/2
    fraction_matched=sum(matched)/denom
    fraction_active=sum(active)/denom
    return {
        "site_sample_count":denom,
        "fraction_matched_within_1pct":fraction_matched,
        "fraction_with_positive_shared_dose":fraction_active,
        "median_fraction_full_left_dose_preserved":median,
        "mean_fraction_full_left_dose_preserved":sum(ratio)/denom,
        "mean_common_direct_dose":sum(t)/denom,
        "relative_dose_gate_pass":fraction_matched>=0.99,
        "minimum_signal_gate_pass":
            fraction_active>=MIN_ACTIVE_FRAC and median>=MIN_MEDIAN_RETAINED,
        "candidate_physically_comparable_and_signal_sufficient":
            (fraction_matched>=0.99 and fraction_active>=MIN_ACTIVE_FRAC
             and median>=MIN_MEDIAN_RETAINED),
        "comparison_is_development_only":True,
        "independent_source_context_count":48,
    }


def run_one(root: Path,out: Path,seed: int,task: str) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model,catalog,source,identity,roster,roster_sha=frozen_model(root,seed,task)
    prep_dir=root/"nec03d"/f"seed{seed}_{task}"
    prepared,prepare_sha,sae,mean,scale=_load_prepared(prep_dir,identity,roster_sha)
    ranked={x["candidate_id"]:x for x in prepared["sae_rankings"]}
    if set(ranked)!={x["id"] for x in roster["candidates"]}:
        raise AssertionError("source-selected candidates changed")
    model_hash=model_digest(model)
    out.mkdir(parents=True,exist_ok=True)
    traces=[]
    summary=[]
    for candidate in roster["candidates"]:
        cid=candidate["id"]
        if cid not in ranked:
            raise AssertionError("candidate missing source-trained SAE features")
        for left,right in PAIRS:
            pair=(left,right)
            effects=[]
            dose_contexts=[]
            errors={"hubs":0,"left":0,"right":0,"clean":0}
            for k in range(48):
                group=catalog["heldout"][16*k:16*(k+1)]
                bits=as_tensor(group)
                labels=target(bits,task)
                with torch.no_grad():
                    clean=model.logits(bits)
                activations=clean_activations(model,bits)
                donors,doses,masks=make_pair(
                    model,activations,candidate,ranked[cid],
                    roster["graph"]["hubs"],sae,mean,scale,pair)
                hubs=graft_forward(model,bits,donors,masks,"hubs")
                logits_l=graft_forward(model,bits,donors,masks,"native")
                logits_r=graft_forward(model,bits,donors,masks,"sham")
                for name,logits in (("clean",clean),("hubs",hubs),
                                    ("left",logits_l),("right",logits_r)):
                    errors[name]+=int((logits.argmax(1)!=labels).sum())
                diff=float(((logits_l.argmax(1)==labels).float()-
                            (logits_r.argmax(1)==labels).float()).mean())
                effects.append(diff)
                dose_contexts.append(doses)
                traces.append({
                    "seed":seed,"task":task,
                    "candidate_id":cid,
                    "comparison":[left,right],
                    "context_bits":list(group[0][4:]),
                    "focal_mask_order":list(range(16)),
                    "labels":labels.tolist(),
                    "clean_logits":clean.double().tolist(),
                    "hubs_logits":hubs.double().tolist(),
                    "left_logits":logits_l.double().tolist(),
                    "right_logits":logits_r.double().tolist(),
                    "site_doses":{str(v):d for v,d in doses.items()},
                })
            doses=pair_dose_stats(dose_contexts)
            summary.append({
                "candidate_id":cid,"pair":[left,right],
                "model_clean_errors_of_768":errors["clean"],
                "hubs_only_errors_of_768":errors["hubs"],
                "left_errors_of_768":errors["left"],
                "right_errors_of_768":errors["right"],
                "left_minus_right_correct_predictions":
                    errors["right"]-errors["left"],
                "left_conditional_rescues_vs_hubs":
                    errors["hubs"]-errors["left"],
                "right_conditional_rescues_vs_hubs":
                    errors["hubs"]-errors["right"],
                "paired_context_left_minus_right_accuracy":
                    interval(effects,seed=3103),
                "achievable_dose":doses,
            })
    if model_digest(model)!=model_hash:
        raise AssertionError("frozen native checkpoint mutated")
    raw=b"".join(serialized(row) for row in traces)
    (out/"pairwise_raw_contexts.jsonl").write_bytes(raw)
    result={
        "schema":SCHEMA,
        "status":"posthoc-attenuation-dose-development",
        "seed":seed,"task":task,
        "model_identity":identity,
        "source_sha256":source["sha256"],
        "original_source_candidate_sha256":roster_sha,
        "source_only_sae_prepare_sha256":prepare_sha,
        "sae_checkpoint_sha256":prepared["sae"]["sae_checkpoint_sha256"],
        "intervention_scope":"same clean-input donors; recurrent memory GELU, hubs-only baseline",
        "dose_policies":{
            "pair_order":[list(x) for x in PAIRS],
            "shared_dose":"min(left_direct,right_direct) mlp_down L2",
            "only_attenuation_no_amplification":True,
            "match_error_tolerance":DOSE_MATCH_TOLERANCE,
            "fraction_matching_required":0.99,
            "required_nonzero_fraction":MIN_ACTIVE_FRAC,
            "required_median_fraction_original_left_dose":MIN_MEDIAN_RETAINED,
            "controls_never_scale_by_outcomes":True,
        },
        "independent_contexts_per_fixed_model":48,
        "focal_variants_per_context":16,
        "candidates":len(roster["candidates"]),
        "comparisons":len(summary),
        "raw_records":len(traces),
        "raw_filename":"pairwise_raw_contexts.jsonl",
        "raw_sha256":hashlib.sha256(raw).hexdigest(),
        "results":summary,
        "no_prospective_significance_inferred":True,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "heldout_exposure":"original PEF-07/NEC-02B context family already examined",
    }
    (out/"pairwise_report.json").write_bytes(serialized(result))
    print(json.dumps({"stage":"NEC-03E","seed":seed,"task":task,
                      "candidate_count":result["candidates"],
                      "comparison_count":result["comparisons"],
                      "raw_records":result["raw_records"],
                      "dose_comparable_and_signal_preserving":sum(
                          x["achievable_dose"]["candidate_physically_comparable_and_signal_sufficient"]
                          for x in summary)},sort_keys=True))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,choices=(103,131,167),default=103)
    p.add_argument("--task",choices=("coalition","threshold"),default="coalition")
    a=p.parse_args(argv)
    return run_one(a.root,a.out,a.seed,a.task)


if __name__=="__main__":
    main()
