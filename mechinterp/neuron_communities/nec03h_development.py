"""NEC-03H-v2: fresh development-world source-only fallback and TopK16 replay.

This version supersedes G only for UNUSED dev-008..015. The original
NEC-03G 0/8 roots and their 450-step training protocols remain intact.
900-step trained checkpoints are freshly initialized with identical
optimizer/data streams; no confirmation world or reserved context is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.neuron_communities.core import build_graph,discover,decode
from mechinterp.neuron_communities.nec03_development import hub_disjoint_sham
from mechinterp.neuron_communities.nec03f_active import build_pair
from mechinterp.neuron_communities.nec03e_pairs import pair_dose_stats
from mechinterp.neuron_communities.matched_dose import graft_forward
from mechinterp.neuron_communities.pef07_adapter import context_attributions
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.topk_sae import fit_topk,dev_ranks
from mechinterp.neuron_communities.nec03g_worlds import (
    DEV_COUNT, MIN_CLEAN_ACC, PRIMARY_MEANINGFUL_DELTA, SCHEMA,
    additive_control_truth, context_catalog, digest, encoded, registry,
    target_bits, worlds,
)
from research.pef07.pef07 import Looper, as_tensor, model_digest
from mechinterp.neuron_communities.nec03h_engine import (
    H_SCHEMA, MODEL_STEPS, SAE_STEPS, train_native_900,
    source_site_selection, training_world,
)
SCHEMA = H_SCHEMA

PAIRS=(("native","sham"),("native","topk"),("topk","topk_alt"))
ROSTER_CONTEXTS=80
GRAPH_MIN_PHI=0.12
GRAPH_MIN_JOINT=3
MAX_COMMUNITY_SIZE=12
MAX_COMMUNITIES_INSPECTED=8
SAE_TOPK=16


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def train_one(world, catalog, root, *, arm, steps=None):
    """Use fixed new-from-seed 900-step source-only training stream."""
    return train_native_900(
        world,catalog,root,arm=arm,
        steps=MODEL_STEPS if steps is None else steps)


@torch.no_grad()
def clean_score(model, rows, world, *, arm):
    correct=0
    truth=0
    for start in range(0,len(rows),16*8):
        bits=as_tensor(rows[start:start+16*8])
        y=target_bits(bits,world,control=arm=="additive")
        logits=model.logits(bits)
        correct+=int((logits.argmax(1)==y).sum())
        truth+=len(y)
    return {"correct":correct,"total":truth,"accuracy":correct/truth}


def source_community(model,catalog,world):
    """Freeze source-graph candidate, or focal-visit non-hub fallback."""
    return source_site_selection(model,catalog,world)


def run_arm(world: dict, catalog: dict, source: dict, out: Path,
            *, arm: str, steps: int, sae_steps: int):
    model,trained=train_one(world,catalog,out,arm=arm,steps=steps)
    native_hash=model_digest(model)
    # First freeze the source-only graph, before any development evaluation.
    candidate,graph,source_discovery=source_community(model,catalog,world)
    graph_bytes=encoded({"world_sha256":world["world_sha256"],
                         "model_sha256":native_hash,
                         "source_sha256":source["sha256"],
                         "arm":arm,"source_discovery":source_discovery})
    (out/f"{arm}.community_source_freeze.json").write_bytes(graph_bytes)
    graph_sha=sha_bytes(graph_bytes)
    if candidate is None:
        score=clean_score(model,catalog["development"],world,arm=arm)
        report={"arm":arm,"train":trained,"clean":score,
                "source_graph_freeze_sha256":graph_sha,
                "source_discovery":source_discovery,
                "candidate_status":"NO_SOURCE_COMMUNITY",
                "source_site_kind":"none",
                "pairs":[],"primary_development_effect":"0",
                "source_manifest_sha256":source["sha256"],
                "raw_paired_context_file":None,
                "raw_paired_context_sha256":None,
                "raw_paired_records":0,
                "world_sha256":world["world_sha256"]}
        (out/f"{arm}.report.json").write_bytes(encoded(report))
        return report
    model_path=out/f"{arm}.sae.pt"
    sae,sae_receipt,mean,scale=fit_topk(
        model,catalog["train"][:80*16],seed=trained["training_seed"],steps=sae_steps)
    ranks,rank_sha=dev_ranks(model,sae,mean,scale,
                             catalog["train"][80*16:160*16])
    torch.save({"state":sae.state_dict(),"mean":mean,"scale":scale,
                "source_contexts_fit":source["contexts_ordered"][:80],
                "source_contexts_rank":source["contexts_ordered"][80:160],
                "model_sha256":native_hash},model_path)
    sae_receipt=dict(sae_receipt)
    sae_receipt["source_sae_sha256"]=sha_file(model_path)
    sae_receipt["source_feature_rank_sha256"]=rank_sha
    sae_receipt["source_fit_contexts"]=80
    sae_receipt["source_rank_contexts"]=80
    sae_receipt["test_and_reserved_contexts_used"]=0
    policy={"hub_disjoint_native_sham":hub_disjoint_sham(candidate,graph)}
    # Frozen policy/SAE/graph receipt goes to disk *before* opening eval.
    freeze={
        "schema":SCHEMA,"world_sha256":world["world_sha256"],
        "model_sha256":native_hash,
        "source_manifest_sha256":source["sha256"],
        "source_graph_candidate":candidate,"graph_hubs":graph["hubs"],
        "graph_full_sha256":digest(graph),
        "source_sae":sae_receipt,
        "source_ranking_per_visit":ranks,
        "native_sham_policy":policy,
        "no_eval_or_reserved_contexts_used_for_selection":True}
    frozen_raw=encoded(freeze)
    freeze_path=out/f"{arm}.source_freeze.json"
    freeze_path.write_bytes(frozen_raw)
    frozen_sha=sha_bytes(frozen_raw)
    score=clean_score(model,catalog["development"],world,arm=arm)
    # Measure every pair, including dose-infeasible and negative endpoints.
    effects=[]
    raw=[]
    for pair in PAIRS:
        errors={"clean":0,"hubs":0,"left":0,"right":0}
        context_effects=[]
        dosage=[]
        for group_index in range(48):
            quartet=catalog["development"][16*group_index:16*(group_index+1)]
            bits=as_tensor(quartet)
            y=target_bits(bits,world,control=arm=="additive")
            with torch.no_grad():
                z=model.logits(bits)
            act=clean_activations(model,bits)
            donors,doses,hubs,selected=__import__(
                "mechinterp.neuron_communities.nec03f_active",
                fromlist=["build_pair"]).build_pair(
                    model,act,candidate,policy,ranks,sae,mean,scale,
                    graph["hubs"],pair)
            with torch.no_grad():
                h=graft_forward(model,bits,donors,hubs,"hubs")
                left=graft_forward(model,bits,donors,hubs,"native")
                right=graft_forward(model,bits,donors,hubs,"sham")
            for name,logit in (("clean",z),("hubs",h),
                               ("left",left),("right",right)):
                errors[name]+=int((logit.argmax(1)!=y).sum())
            context_effects.append(float(
                ((left.argmax(1)==y).float()-(right.argmax(1)==y).float()).mean()))
            dosage.append(doses)
            context_positions=source["context_positions"]
            raw.append({"world_id":world["world_id"],"arm":arm,
                        "pair":list(pair),"source_context_index":group_index,
                        "source_context_bits":[quartet[0][k] for k in context_positions],
                        "focal_mask_order":list(range(16)),
                        "labels":y.tolist(),
                        "clean_logits":z.detach().double().tolist(),
                        "hubs_logits":h.double().tolist(),
                        "left_logits":left.double().tolist(),
                        "right_logits":right.double().tolist(),
                        "local_physical_doses":{str(k):v for k,v in doses.items()},
                        "active_features":{str(k):v for k,v in selected.items()}})
        site=pair_dose_stats(dosage)
        effects.append({
            "pair":list(pair),
            "errors_of_768":errors,
            "left_minus_right_correct":errors["right"]-errors["left"],
            "left_minus_right_accuracy":(errors["right"]-errors["left"])/768,
            "physical_dose":site,
            "paired_context_differences":context_effects,
        })
    raw_bytes=b"".join(encoded(r) for r in raw)
    filename=f"{arm}.raw_pairs.jsonl"
    (out/filename).write_bytes(raw_bytes)
    if model_digest(model)!=native_hash:
        raise AssertionError("frozen PEF-07 native weights changed during causal replay")
    report={
        "arm":arm,"train":trained,"clean":score,
        "source_discovery":source_discovery,
        "candidate_status":"SOURCE_SELECTED",
        "source_site_kind":"fallback" if source_discovery["fallback_used"] else "graph",
        "source_freeze_sha256":frozen_sha,
        "source_graph_freeze_sha256":graph_sha,
        "source_only_sae":sae_receipt,
        "source_manifest_sha256":source["sha256"],
        "world_sha256":world["world_sha256"],
        "pairs":effects,
        "raw_paired_context_file":filename,
        "raw_paired_context_sha256":sha_bytes(raw_bytes),
        "raw_paired_records":len(raw),
        "development_contexts":48,
        "reserved_contexts_opened":0,
        "all_selection_frozen_before_evaluation":True}
    (out/f"{arm}.report.json").write_bytes(encoded(report))
    return report


def root_effects(world, reports):
    """V2 ITT: composed competence + primary-site + primary physical dose.

    An incompetent or unavailable ADDITIVE control invalidates its SHAM,
    while the composed primary remains reportable. A missing additive null
    receives an explicitly conservative effect magnitude 1 in the
    preflight, and is permanently flagged as uncalibrated.
    """
    composed=reports["composed"]
    additive=reports["additive"]
    reasons=[]
    if composed["clean"]["accuracy"] < MIN_CLEAN_ACC:
        reasons.append("composed_incompetent_clean_head")
    if composed["candidate_status"]!="SOURCE_SELECTED":
        reasons.append("composed_no_source_site")
    composed_pairs={}
    if composed["candidate_status"]=="SOURCE_SELECTED":
        composed_pairs={tuple(x["pair"]):x for x in composed["pairs"]}
        if ("topk","topk_alt") not in composed_pairs:
            raise AssertionError("missing v2 primary pair")
        if not composed_pairs[("topk","topk_alt")]["physical_dose"][
                "candidate_physically_comparable_and_signal_sufficient"]:
            reasons.append("composed_primary_insufficient_physical_signal")
    primary=(composed_pairs.get(("topk","topk_alt"),{})
             .get("left_minus_right_accuracy",None))
    additive_valid=(additive["clean"]["accuracy"]>=MIN_CLEAN_ACC
                    and additive["candidate_status"]=="SOURCE_SELECTED")
    additive_pairs={}
    if additive["candidate_status"]=="SOURCE_SELECTED":
        additive_pairs={tuple(x["pair"]):x for x in additive["pairs"]}
        additive_valid=additive_valid and bool(
            additive_pairs[("topk","topk_alt")]["physical_dose"][
                "candidate_physically_comparable_and_signal_sufficient"])
    # All zero-eligible control effects are documented; a fake zero
    # sham would bias the upper null floor downward, so use a maximally
    # conservative signed bound of 1 when source control is missing.
    additive_sham=additive_pairs[("topk","topk_alt")]["left_minus_right_accuracy"] if additive_valid else 1.0
    native_record=composed_pairs.get(("native","sham"))
    if native_record and native_record["physical_dose"][
            "candidate_physically_comparable_and_signal_sufficient"]:
        native_sham=native_record["left_minus_right_accuracy"]
        native_valid=True
    else:
        native_sham=1.0
        native_valid=False
    primary_valid=not reasons
    return {
        "root_primary_effect":str(round(primary,12)) if primary_valid else "0",
        "primary_observed_if_any":primary,
        "sham_effects":[str(round(additive_sham,12)),
                        str(round(native_sham,12))],
        "composed_clean_eligible":composed["clean"]["accuracy"]>=MIN_CLEAN_ACC,
        "composed_source_site_kind":composed["source_site_kind"],
        "additive_source_site_kind":additive["source_site_kind"],
        "additive_sham_valid":additive_valid,
        "native_sham_valid":native_valid,
        "null_floor_calibration_eligible":primary_valid and additive_valid and native_valid,
        "unmeasured_shams_conservative_bound":1.0,
        "planned_world_valid":primary_valid,
        "failure_reasons":reasons,
        "control_failures":[
            x for x,valid in (("additive_sham_unavailable",additive_valid),
                               ("native_sham_unavailable",native_valid))
            if not valid],
        "effect_class":"composed-topk-versus-active-alternative-development-v2",
        "description":"Original zero ITT when composed primary fails; auxiliary sham failures remain visible and conservatively bounded."
    }


def run(world_index: int, out: Path, *, trunk_steps: int=MODEL_STEPS,
        sae_steps: int=SAE_STEPS):
    """Only unused dev-008..015 are eligible for full V2 measurement."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    world=training_world(world_index,calib=False)
    cohort=registry()
    if cohort["dev_full_world_specs"][world_index]!=world:
        raise AssertionError("source world registry changed")
    out.mkdir(parents=True,exist_ok=True)
    catalog,source=context_catalog(world)
    (out/"context_source_manifest.json").write_bytes(encoded(source))
    (out/"world_spec.json").write_bytes(encoded(world))
    arms={}
    for arm in ("composed","additive"):
        arms[arm]=run_arm(world,catalog,source,out,arm=arm,
                          steps=trunk_steps,sae_steps=sae_steps)
    effects=root_effects(world,arms)
    full=(trunk_steps==MODEL_STEPS and sae_steps==SAE_STEPS)
    if not full:
        effects["effect_class"]="engineering-short-run-not-for-NEC03H-preflight"
    report={
        "schema":H_SCHEMA,"world_id":world["world_id"],
        "world_index":world_index,"world_sha256":world["world_sha256"],
        "cohort":"development",
        "original_NEC03G_world_registry_sha256":cohort["registry_sha256"],
        "world_source_sha256":source["sha256"],
        "trunk_steps":trunk_steps,"sae_steps":sae_steps,
        "full_registered_v2_recipe":full,
        "models_trained_from_original_fresh_seeds":True,
        "teacher_models_independently_initialized":True,
        "source_sites_frozen_before_development_readout":True,
        "original_NEC03G_0_8_results_unchanged":True,
        "original_confirmation_worlds_evaluated":0,
        "reserved_context_groups_evaluated":0,
        "native_clean_scores":{arm:z["clean"] for arm,z in arms.items()},
        "source_site_kinds":{arm:z["source_site_kind"] for arm,z in arms.items()},
        "candidate_source_status":{
            arm:z["candidate_status"] for arm,z in arms.items()},
        "primary_effect":effects,
        "independent_world_id":world["world_id"],
        "failed_composed_roots_included_in_ITT":True,
        "FSI_01_class":"DEVELOPMENT_PILOT_ONLY",
    }
    (out/"root_summary.json").write_bytes(encoded(report))
    print(json.dumps({"world":world["world_id"],"full":full,
                      "native_accuracy":{
                          a:z["clean"]["correct"] for a,z in arms.items()},
                      "site_kinds":report["source_site_kinds"],
                      "root_primary_effect":effects["root_primary_effect"],
                      "primary_eligible":effects["planned_world_valid"],
                      "sham_valid":effects["null_floor_calibration_eligible"],
                      "failed":effects["failure_reasons"],
                      "confirmation_worlds":0},sort_keys=True),flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development-world-index",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--trunk-steps",type=int,default=MODEL_STEPS)
    p.add_argument("--sae-steps",type=int,default=SAE_STEPS)
    args=p.parse_args(argv)
    run(args.development_world_index,args.out,
        trunk_steps=args.trunk_steps,sae_steps=args.sae_steps)


if __name__=="__main__":
    main()
