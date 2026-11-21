"""NEC-03F TopK16 SAE with input-conditioned active source-feature routing.

Source 80/80 original parameter-training context partition freezes the SAE,
ranking and hub-disjoint sham before any PEF-07 heldout replay. All training
and causal validation are retrospective development on previously seen roots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from mechinterp.neuron_communities.core import decode, interval, node
from mechinterp.neuron_communities.matched_dose import (
    by_visit, masked, graft_forward,
)
from mechinterp.neuron_communities.nec03_development import (
    frozen_model, hub_disjoint_sham, serialized, source_ids,
)
from mechinterp.neuron_communities.nec03e_pairs import (
    attenuate_pair,pair_dose_stats,
)
from mechinterp.neuron_communities.source_sae import (
    TRAIN_CONTEXTS, DEV_CONTEXTS, clean_activations,
)
from mechinterp.neuron_communities.topk_sae import (
    TOP_K,TopKSAE,fit_topk,dev_ranks,active_feature_policy,
)
from research.pef07.pef07 import as_tensor,model_digest,target

SCHEMA="NEC-03F/source-TopK16-dynamic-active-feature-pairwise-dose/v1"
PAIRS=(("native","sham"),("native","topk"),("topk","topk_alt"))
MIN_ACTIVE_FRAC=0.80
MIN_RETAINED_MEDIAN=0.25


def prepare(root: Path,out: Path,seed: int,task: str) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model,catalog,source,identity,roster,roster_sha=frozen_model(root,seed,task)
    before=model_digest(model)
    train=catalog["train"][:16*TRAIN_CONTEXTS]
    dev=catalog["train"][16*TRAIN_CONTEXTS:16*(TRAIN_CONTEXTS+DEV_CONTEXTS)]
    train_id,dev_id=source_ids(train),source_ids(dev)
    if len(set(train_id))!=80 or len(set(dev_id))!=80 or set(train_id)&set(dev_id):
        raise AssertionError("TopK source split overlap")
    sae,receipt,mean,scale=fit_topk(model,train,seed=seed)
    ranks,rank_sha=dev_ranks(model,sae,mean,scale,dev)
    out.mkdir(parents=True,exist_ok=True)
    path=out/"source_topk16.pt"
    torch.save({"state":sae.state_dict(),"mean":mean,"scale":scale,
                "receipt":receipt,
                "source_context_ids_train":train_id,
                "source_context_ids_rank":dev_id},path)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    candidates=[{"id":x["id"],"native_nodes":x["nodes"],
                 "hub_disjoint_native_sham":hub_disjoint_sham(x,roster["graph"])}
                for x in roster["candidates"]]
    manifest={
        "schema":SCHEMA,
        "stage":"source-only-topk16-training-and-routing-frozen",
        "seed":seed,"task":task,
        "frozen_model_identity":identity,
        "original_discovery_sha256":roster_sha,
        "source_manifest_sha256":source["sha256"],
        "train_only_contexts":{"fit":80,"rank":80,"heldout":0},
        "train_context_ids":train_id,
        "rank_context_ids":dev_id,
        "topk_checkpoint_file":path.name,
        "topk_checkpoint_sha256":sha,
        "topk_training":receipt,
        "source_rank_sha256":rank_sha,
        "source_feature_rank_per_visit":ranks,
        "native_community_source_candidates":candidates,
        "routing_policy":{
            "name":"top-and-bottom-source-attribution-ranked-active-TopK-feature",
            "topk_active_per_input":TOP_K,
            "selected_feature_count_per_visit_equals_native_community":True,
            "rank_fitted_from_train_only":True,
            "per_input_policy_uses_activations_only":True,
            "teacher_labels_visible_to_selector":False,
        },
        "inference":"development-only-on-previously-exposed-contexts",
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    raw=serialized(manifest)
    (out/"prepare.json").write_bytes(raw)
    (out/"prepare.sha256").write_text(hashlib.sha256(raw).hexdigest()+"\n")
    if model_digest(model)!=before:
        raise AssertionError("TopK preparation changed frozen PEF weights")
    print(json.dumps({"stage":"prepare","seed":seed,"task":task,
                      "candidate_count":len(candidates),
                      "reconstruction_mse":receipt["train_normalized_reconstruction_mse"],
                      "mean_l0":receipt["observed_mean_active_features_per_input"],
                      "sae_sha256":sha},sort_keys=True))
    return manifest


def load_prepared(out,identity,roster_sha):
    raw=(out/"prepare.json").read_bytes()
    checksum=hashlib.sha256(raw).hexdigest()
    if checksum!=(out/"prepare.sha256").read_text().strip():
        raise AssertionError("source-ranked TopK feature policy mutated")
    obj=json.loads(raw)
    if obj["frozen_model_identity"]!=identity or obj["original_discovery_sha256"]!=roster_sha:
        raise AssertionError("wrong PEF model or native candidates for TopK code")
    if obj["train_only_contexts"]!={"fit":80,"rank":80,"heldout":0}:
        raise AssertionError("source TopK training leaked heldout")
    path=out/obj["topk_checkpoint_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=obj["topk_checkpoint_sha256"]:
        raise AssertionError("frozen source TopK checkpoint bytes changed")
    data=torch.load(path,map_location="cpu",weights_only=True)
    if (data["receipt"]["model_sha256"]!=identity["model_sha256"]
        or data["source_context_ids_train"]!=obj["train_context_ids"]
        or data["source_context_ids_rank"]!=obj["rank_context_ids"]):
        raise AssertionError("TopK train/feature source checkpoint differs")
    sae=TopKSAE()
    sae.load_state_dict(data["state"],strict=True)
    sae.eval()
    return obj,checksum,sae,data["mean"],data["scale"]


@torch.no_grad()
def build_pair(model,activations,candidate,policy,rank,sae,mean,scale,hubs,pair):
    if pair not in PAIRS:
        raise ValueError("unsupported pair")
    chosen=by_visit(candidate["nodes"])
    shams=by_visit(policy["hub_disjoint_native_sham"])
    hubs_by=by_visit(hubs)
    donors={}
    doses={}
    masks={}
    selections={}
    w=model.block.mlp_down.weight.detach()
    for visit,ind in sorted(chosen.items()):
        if len(ind)!=len(shams.get(visit,set())):
            raise AssertionError("native shams differ in site cardinality")
        if ind & hubs_by.get(visit,set()) or shams[visit] & hubs_by.get(visit,set()):
            raise AssertionError("source selected sham/native overlaps hubs")
        a=activations[:,visit,:].detach()
        top,alt,choice=active_feature_policy(
            sae,mean,scale,a,rank[visit],n=len(ind))
        base={
            "native":masked(a,ind),
            "sham":masked(a,shams[visit]),
            "topk":top,
            "topk_alt":alt,
        }
        left,right,receipt=attenuate_pair(base[pair[0]],base[pair[1]],w)
        donors[visit]={"hubs":torch.zeros_like(left),
                       "native":left,"sham":right}
        doses[visit]=receipt
        selections[visit]=choice
        masks[visit]=masked(torch.ones_like(a),hubs_by.get(visit,set()))[0]
    return donors,doses,masks,selections


def evaluate(root: Path,out: Path,seed: int,task: str) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model,catalog,source,identity,roster,roster_sha=frozen_model(root,seed,task)
    prep,prep_sha,sae,mean,scale=load_prepared(out,identity,roster_sha)
    if [x["id"] for x in prep["native_community_source_candidates"]] != [
            x["id"] for x in roster["candidates"]]:
        raise AssertionError("source-only TopK candidate list differs")
    policies={x["id"]:x for x in prep["native_community_source_candidates"]}
    ranks=prep["source_feature_rank_per_visit"]
    before=model_digest(model)
    rows=[]
    results=[]
    for cand in roster["candidates"]:
        cid=cand["id"]
        for pair in PAIRS:
            errors={"clean":0,"hubs":0,"left":0,"right":0}
            context_effects=[]
            dose_contexts=[]
            for context in range(48):
                group=catalog["heldout"][16*context:16*(context+1)]
                bits=as_tensor(group)
                labels=target(bits,task)
                with torch.no_grad():
                    clean=model.logits(bits)
                a=clean_activations(model,bits)
                donors,doses,masks,selection=build_pair(
                    model,a,cand,policies[cid],ranks,sae,mean,scale,
                    roster["graph"]["hubs"],pair)
                hubs=graft_forward(model,bits,donors,masks,"hubs")
                left=graft_forward(model,bits,donors,masks,"native")
                right=graft_forward(model,bits,donors,masks,"sham")
                for key,logits in (("clean",clean),("hubs",hubs),
                                   ("left",left),("right",right)):
                    errors[key]+=int((logits.argmax(1)!=labels).sum())
                context_effects.append(float(((left.argmax(1)==labels).float()-
                                               (right.argmax(1)==labels).float()).mean()))
                dose_contexts.append(doses)
                rows.append({
                    "seed":seed,"task":task,"candidate_id":cid,
                    "pair":list(pair),"context_bits":list(group[0][4:]),
                    "mask_order":list(range(16)),"labels":labels.tolist(),
                    "clean_logits":clean.double().tolist(),
                    "hubs_logits":hubs.double().tolist(),
                    "left_logits":left.double().tolist(),
                    "right_logits":right.double().tolist(),
                    "site_doses":{str(k):v for k,v in doses.items()},
                    "site_active_policy":{str(k):v for k,v in selection.items()},
                })
            dose=pair_dose_stats(dose_contexts)
            results.append({
                "candidate_id":cid,"pair":list(pair),
                "clean_errors_of_768":errors["clean"],
                "hubs_errors_of_768":errors["hubs"],
                "left_errors_of_768":errors["left"],
                "right_errors_of_768":errors["right"],
                "left_minus_right_correct_count":errors["right"]-errors["left"],
                "left_rescued_vs_hubs":errors["hubs"]-errors["left"],
                "right_rescued_vs_hubs":errors["hubs"]-errors["right"],
                "source_context_paired_accuracy_difference":
                    interval(context_effects,seed=3901),
                "physical_dose":dose})
    if model_digest(model)!=before:
        raise AssertionError("frozen model altered by TopK evaluation")
    trace=b"".join(serialized(row) for row in rows)
    (out/"raw_context_pairs.jsonl").write_bytes(trace)
    report={
        "schema":SCHEMA,
        "stage":"posthoc-active-topk-donor-evaluation",
        "seed":seed,"task":task,
        "frozen_model_identity":identity,
        "source_manifest_sha256":source["sha256"],
        "source_topk_prepare_sha256":prep_sha,
        "original_native_discovery_sha256":roster_sha,
        "topk_checkpoint_sha256":prep["topk_checkpoint_sha256"],
        "candidate_count":len(roster["candidates"]),
        "comparison_count":len(results),
        "source_contexts_heldout_per_model":48,
        "focal_mask_variants_per_context":16,
        "pair_names":[list(p) for p in PAIRS],
        "dose_policy":"attenuation-only shared min frozen mlp_down L2 per input and visit",
        "topk_feature_policy":"source-ranked top/bottom active input-conditional features",
        "matched_within_1pct_required":0.99,
        "minimum_positive_dose_fraction":MIN_ACTIVE_FRAC,
        "minimum_median_retained_dose":MIN_RETAINED_MEDIAN,
        "raw_file":"raw_context_pairs.jsonl",
        "raw_sha256":hashlib.sha256(trace).hexdigest(),
        "raw_records":len(rows),
        "results":results,
        "numerically_dose_and_signal_feasible_pairs":sum(
            r["physical_dose"]["candidate_physically_comparable_and_signal_sufficient"]
            for r in results),
        "no_claim_of_out_of_family_generalization":True,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"report.json").write_bytes(serialized(report))
    print(json.dumps({"stage":"eval","seed":seed,"task":task,
                      "candidates":report["candidate_count"],
                      "records":report["raw_records"],
                      "feasible":report["numerically_dose_and_signal_feasible_pairs"]},
                      sort_keys=True))
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
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
