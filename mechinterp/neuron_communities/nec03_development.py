"""NEC-03D: source-only SAE fitting and posthoc matched-dose causal replay.

Two explicit phases prevent training or feature selection from seeing
PEF-07 heldout contexts. This is a development study on previously opened
PEF-07 source families, never confirmation for NEC-03 issue #239.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from mechinterp.neuron_communities.core import decode, node, interval, summary
from mechinterp.neuron_communities.matched_dose import (
    MODES, donor_roster, dose_summary, graft_forward,
)
from mechinterp.neuron_communities.pef07_adapter import (
    source_model, paired_effect,
)
from mechinterp.neuron_communities.source_sae import (
    DEV_CONTEXTS, FEATURES, STEPS, TRAIN_CONTEXTS, SourceSAE, clean_activations,
    dev_feature_importance, feature_rosters, fit_source_sae, save_sae, seal_array,
)
from research.pef07.pef07 import as_tensor, model_digest, target

EXPECTED_SHA = "de4e0263864ee70c2bf94ca95008ace10fce1323d1a51b4c8829ec26e40bbd9b"
SCHEMA = "NEC-03D/source-only-sae-physical-dose-exploratory/v1"


def serialized(obj: dict) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"))+"\n").encode()


def read_roster(root: Path, *, seed: int, task: str):
    folder=root/"analysis"/f"seed{seed}_{task}"
    raw=(folder/"discovery.json").read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    if (folder/"discovery.sha256").read_text().strip()!=sha:
        raise AssertionError("NEC-02B candidate roster changed")
    roster=json.loads(raw)
    if roster["seed"]!=seed or roster["task"]!=task:
        raise AssertionError("incorrect frozen candidate source")
    if roster["source_split"]!="train_only" or roster["context_groups_used"]!=160:
        raise AssertionError("candidate source was not train-only")
    return roster,sha


def frozen_model(root: Path,seed: int,task: str):
    args=SimpleNamespace(smoke=False,seed=seed,task=task,
                         checkpoint_dir=root/"pef07")
    model,catalog,source,identity=source_model(args)
    if source["sha256"]!=EXPECTED_SHA:
        raise AssertionError("unexpected PEF-07 source split")
    roster,sha=read_roster(root,seed=seed,task=task)
    if roster["identity"]!=identity:
        raise AssertionError("frozen candidate/model identity mismatch")
    return model,catalog,source,identity,roster,sha


def source_ids(rows) -> list[str]:
    if len(rows)%16:
        raise AssertionError("incomplete train context")
    return ["".join(str(x) for x in rows[i][4:])
            for i in range(0,len(rows),16)]


def hub_disjoint_sham(candidate: dict, graph: dict) -> list[str]:
    """Repair old train-frequency shams which overlap hubs.

    NEC-01 allowed some same-frequency sham nodes to coincide with hubs.
    An additive graft would double count those hubs. Refit entirely
    on train-only graph frequencies and freeze before evaluation.
    """
    from statistics import mean
    per_visit={}
    for key in candidate["nodes"]:
        v,j=decode(key)
        per_visit.setdefault(v,[]).append(j)
    blocked=set(candidate["nodes"]) | set(graph["hubs"])
    freq=graph["frequency"]
    selected=[]
    for visit,wanted in sorted(per_visit.items()):
        target=mean(freq.get(node(visit,j),0) for j in wanted)
        options=[node(visit,j) for j in range(64)
                 if node(visit,j) not in blocked]
        options.sort(key=lambda k:(abs(freq.get(k,0)-target),decode(k)[1]))
        if len(options)<len(wanted):
            raise AssertionError("source graph lacks hub-disjoint sham units")
        selected.extend(options[:len(wanted)])
    if set(selected) & blocked:
        raise AssertionError("matched sham collided with hubs/native")
    return selected


def prepare(root: Path,out: Path,seed: int,task: str) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model,catalog,src,identity,roster,roster_sha=frozen_model(root,seed,task)
    original=model_digest(model)
    train=catalog["train"][:16*TRAIN_CONTEXTS]
    dev=catalog["train"][16*TRAIN_CONTEXTS:16*(TRAIN_CONTEXTS+DEV_CONTEXTS)]
    if len(train)!=1280 or len(dev)!=1280:
        raise AssertionError("wrong source-only disjoint subsets")
    t_ids,d_ids=source_ids(train),source_ids(dev)
    if len(set(t_ids))!=80 or len(set(d_ids))!=80 or set(t_ids)&set(d_ids):
        raise AssertionError("SAE fit and feature ranking share context identity")
    sources={
        "full_original_source_sha256":src["sha256"],
        "train_context_ids":t_ids,
        "train_dev_context_ids":d_ids,
        "train_contexts":len(t_ids),
        "train_dev_contexts":len(d_ids),
        "heldout_context_ids_used":[],
        "sae_fit_may_read_original_training_split_only":True,
    }
    sae,receipt,mean,scale=fit_source_sae(model,train,seed=seed)
    scores=dev_feature_importance(model,sae,mean,scale,dev)
    if not torch.isfinite(scores).all() or scores.shape!=(12,FEATURES):
        raise AssertionError("feature ranking invalid")
    ranked=[]
    for candidate in roster["candidates"]:
        features,control=feature_rosters(scores,candidate["nodes"],seed=1907)
        ranked.append({"candidate_id":candidate["id"],
                       "native_nodes":candidate["nodes"],
                       "original_nec01_frequency_matched_sham": candidate["frequency_matched_sham"],
                       "hub_disjoint_native_sham":hub_disjoint_sham(candidate,roster["graph"]),
                       "sae_feature_indices_by_visit":
                           {str(k):v for k,v in features.items()},
                       "sae_random_features_by_visit":
                           {str(k):v for k,v in control.items()}})
    out.mkdir(parents=True,exist_ok=True)
    receipt=save_sae(out/"source_sae.pt",sae,receipt,mean,scale,sources=sources)
    if model_digest(model)!=original:
        raise AssertionError("source-only SAE pipeline modified model")
    manifest={
        "schema":SCHEMA,
        "status":"source-only-sae-and-features-frozen",
        "seed":seed,"task":task,
        "source_sha256":src["sha256"],
        "model_identity":identity,
        "original_candidate_roster_sha256":roster_sha,
        "selected_candidate_count":len(ranked),
        "source_contexts":{"fit":80,"selection":80,"heldout":0},
        "sae":receipt,
        "sae_feature_rankings_sha256":seal_array(scores),
        "sae_rankings":ranked,
        "raw_source_contexts":sources,
        "causal_conditions":["hubs","native","sham","sae","sae_sham"],
        "physical_dose_method":"per-example per-visit L2 frozen mlp_down output",
        "physical_dose_gain_cap":8.0,
        "physical_dose_match_tolerance_fraction":0.05,
        "required_fraction_matched_for_comparability":0.90,
        "train_fitted_prior_to_heldout_evaluation":True,
        "heldout_exposure_note":
            "PEF-07/NEC-02B heldout contexts were inspected in earlier studies; this is posthoc development.",
        "inference_gate":"REDESIGN_OR_PILOT_ONLY"
    }
    raw=serialized(manifest)
    (out/"prepare.json").write_bytes(raw)
    (out/"prepare.sha256").write_text(hashlib.sha256(raw).hexdigest()+"\n")
    print(json.dumps({"stage":"prepare","seed":seed,"task":task,
                      "candidates":len(ranked),
                      "recon_mse":receipt["training_reconstruction_mse_normalized"],
                      "mean_l0":receipt["mean_active_features"],
                      "roster_sha256":hashlib.sha256(raw).hexdigest()},sort_keys=True))
    return manifest


def _load_prepared(out: Path, identity, roster_sha):
    raw=(out/"prepare.json").read_bytes()
    checksum=hashlib.sha256(raw).hexdigest()
    if checksum!=(out/"prepare.sha256").read_text().strip():
        raise AssertionError("source SAE feature selection unfrozen")
    manifest=json.loads(raw)
    if manifest["model_identity"]!=identity:
        raise AssertionError("wrong model for SAE")
    if manifest["original_candidate_roster_sha256"]!=roster_sha:
        raise AssertionError("frozen discovery roster changed")
    path=out/manifest["sae"]["sae_checkpoint_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest["sae"]["sae_checkpoint_sha256"]:
        raise AssertionError("source SAE weights changed after freeze")
    data=torch.load(path,map_location="cpu",weights_only=True)
    if data["receipt"]["fitted_trunk_model_sha256"]!=identity["model_sha256"]:
        raise AssertionError("SAE checkpoint trained on another model")
    if data["sources"]!=manifest["raw_source_contexts"]:
        raise AssertionError("source SAE split metadata differs")
    sae=SourceSAE()
    sae.load_state_dict(data["state"],strict=True)
    sae.eval()
    return manifest,checksum,sae,data["mean"],data["scale"]


def _feature_indices(row,name):
    return {int(k):v for k,v in row[name].items()}


def _avg(values):
    return sum(values)/len(values) if values else None


def evaluate(root: Path,out: Path,seed: int,task: str,
             *, context_count: int=48) -> dict:
    torch.set_num_threads(1)
    if context_count!=48:
        raise ValueError("only complete 48-context descriptive replay admitted")
    model,catalog,src,identity,roster,roster_sha=frozen_model(root,seed,task)
    prepared,pre_sha,sae,mean,scale=_load_prepared(out,identity,roster_sha)
    ranks=prepared["sae_rankings"]
    if [x["candidate_id"] for x in ranks]!=[x["id"] for x in roster["candidates"]]:
        raise AssertionError("post-selection roster mismatch")
    original=model_digest(model)
    entries=[]
    results=[]
    for cand, ranked in zip(roster["candidates"],ranks):
        per_mode={name:[] for name in MODES}
        dose_rows=[]
        dif_native_sham=[]
        dif_sae_random=[]
        errors={name:0 for name in MODES}
        clean_errors=0
        for group_idx in range(context_count):
            group=catalog["heldout"][group_idx*16:(group_idx+1)*16]
            if len(group)!=16 or len({row[4:] for row in group})!=1:
                raise AssertionError("incomplete heldout focal quartet")
            bits=as_tensor(group)
            y=target(bits,task)
            with torch.no_grad():
                clean_logits=model.logits(bits)
            act=clean_activations(model,bits)
            donors,doses,hubs=donor_roster(
                model,act,cand["nodes"],roster["graph"]["hubs"],
                ranked["hub_disjoint_native_sham"],
                sae,mean,scale,
                _feature_indices(ranked,"sae_feature_indices_by_visit"),
                _feature_indices(ranked,"sae_random_features_by_visit"))
            logits={}
            metrics={}
            correct_clean=int((clean_logits.argmax(1)==y).sum())
            clean_errors+=16-correct_clean
            for name in MODES:
                changed=graft_forward(model,bits,donors,hubs,name)
                logits[name]=changed.double().tolist()
                effect=paired_effect(clean_logits,changed,y)
                metrics[name]=effect
                per_mode[name].append(effect)
                errors[name]+=int((changed.argmax(1)!=y).sum())
            dif_native_sham.append(
                metrics["sham"]["correctness_drop"]-
                metrics["native"]["correctness_drop"])
            dif_sae_random.append(
                metrics["sae_sham"]["correctness_drop"]-
                metrics["sae"]["correctness_drop"])
            dose_rows.append(doses)
            entries.append({
                "candidate_id":cand["id"],
                "seed":seed,"task":task,
                "context_bits":list(group[0][4:]),
                "counterfactual_mask_order":list(range(16)),
                "labels":y.tolist(),
                "clean_logits":clean_logits.double().tolist(),
                "intervention_logits":logits,
                "local_physical_doses":{str(k):v for k,v in doses.items()},
                "metrics":metrics,
                "clean_causal_donor":True,
            })
        dose={mode:dose_summary(dose_rows,mode)
              for mode in ("sham","sae","sae_sham")}
        contrast={
            "native_minus_matched_sham_accuracy_per_context":
                interval(dif_native_sham,seed=2103),
            "sae_minus_matched_sae_random_accuracy_per_context":
                interval(dif_sae_random,seed=2104)}
        results.append({
            "candidate_id":cand["id"],
            "native_nodes":cand["nodes"],
            "source_selected_sae_features":ranked["sae_feature_indices_by_visit"],
            "source_selected_sae_sham_features":ranked["sae_random_features_by_visit"],
            "clean_errors_of_768":clean_errors,
            "errors_of_768_by_mode":errors,
            "conditional_rescues_vs_hubs_errors_of_768":
                {mode:errors["hubs"]-errors[mode] for mode in MODES if mode!="hubs"},
            "physical_dose_audit":dose,
            "all_dose_controls_meet_90pct_5pct_tolerance":
                all(not x["dose_failure_if_under_90pct_matched"]
                    for x in dose.values()),
            "paired_context_contrasts":contrast,
            "five_condition_context_summaries":
                {name:summary(per_mode[name],seed=2105+i)
                 for i,name in enumerate(MODES)},
            "comparability_note":
                "Clean-donor additive grafts into hubs-only GELU, dose matched at frozen MLP-down output. "
                "Dose-clip failures disqualify dose-comparable claims."})
    raw=b"".join(serialized(row) for row in entries)
    name="raw_matched_graft_contexts.jsonl"
    (out/name).write_bytes(raw)
    result={
        "schema":SCHEMA,
        "status":"posthoc-development-only-causal-outcomes",
        "model_identity":identity,
        "source_sha256":src["sha256"],
        "prepare_sha256":pre_sha,
        "original_candidate_roster_sha256":roster_sha,
        "seed":seed,"task":task,
        "independent_context_groups":context_count,
        "correlated_focal_cases_per_context":16,
        "original_checkpoint_untouched":model_digest(model)==original,
        "original_PEF07_score_changed":False,
        "features_fitted_on_heldout":False,
        "raw_filename":name,
        "raw_sha256":hashlib.sha256(raw).hexdigest(),
        "raw_records":len(entries),
        "candidates":results,
        "familywise_significance_claim":False,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "readout_exposure":
            "All original 48 heldout PEF-07 contexts were opened by NEC-01/02B before this development analysis.",
    }
    if model_digest(model)!=original:
        raise AssertionError("causal experiment mutated frozen model")
    (out/"matched_graft_report.json").write_bytes(serialized(result))
    print(json.dumps({"stage":"evaluate","seed":seed,"task":task,
                      "candidate_count":len(results),
                      "contexts":context_count,
                      "raw_records":len(entries),
                      "all_physically_matched":
                          all(x["all_dose_controls_meet_90pct_5pct_tolerance"]
                              for x in results)},sort_keys=True))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","evaluate"))
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,choices=(103,131,167),default=103)
    p.add_argument("--task",choices=("coalition","threshold"),default="coalition")
    args=p.parse_args(argv)
    if args.stage=="prepare":
        return prepare(args.root,args.out,args.seed,args.task)
    return evaluate(args.root,args.out,args.seed,args.task)


if __name__=="__main__":
    main()
