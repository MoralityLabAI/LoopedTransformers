"""NEC-03I-B: source-rank specificity versus four hash-scrambled ACTIVE shams.

Read-only native and TopK16 SAE checkpoint replay on the eight previously
exposed NEC-03H-v2 development roots. Each source-ranked active SAE donor
is compared with a label-blind, input-conditioned disjoint active-feature
control selected using a frozen cryptographic hash ranking. Four rank
seeds are fixed before this new intervention panel runs.

Physical local mlp_down output L2 dose is equalized by attenuation only.
The 64 confirmation worlds and all reserved context groups are forbidden.
All effects are retrospective development, never independent-world p-values.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import torch

from mechinterp.neuron_communities.matched_dose import (
    by_visit, masked, graft_forward,
)
from mechinterp.neuron_communities.nec03e_pairs import attenuate_pair, pair_dose_stats
from mechinterp.neuron_communities.nec03f_active import build_pair
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.topk_sae import TopKSAE, active_feature_policy
from mechinterp.neuron_communities.nec03g_worlds import context_catalog, registry, target_bits, worlds
from research.pef07.pef07 import Looper, as_tensor, model_digest

SCHEMA = "NEC-03I-B/4-hash-active-rank-shams/v1"
VARIANTS = tuple(range(4))
RANK_SALT = b"NEC03IB-hash-active-shams-v1"
SOURCE_FAMILY = "NEC-03H/source-site-fallback-900steps-development-v2"
FROZEN_ORIGINAL_RUN_ID = 44878411694


def verify_panel_lock(lock_path: Path) -> dict:
    """Proof that the four exploratory sham policies were frozen before replay."""
    cfg=json.loads(lock_path.read_text())
    if cfg.get("schema")!="NEC-03I-B/active-hash-sham-lock/v1":
        raise AssertionError("new control family lacks versioned source lock")
    if (cfg.get("original_run_id")!=FROZEN_ORIGINAL_RUN_ID
        or cfg.get("world_indices")!=list(range(8,16))
        or cfg.get("variants")!=list(VARIANTS)
        or cfg.get("hash_salt")!=RANK_SALT.decode()):
        raise AssertionError("source-control family changed")
    if (cfg.get("confirmation_worlds_scored")!=0
        or cfg.get("reserved_contexts_scored")!=0
        or cfg.get("original_registry_sha256")!=registry()["registry_sha256"]
        or cfg.get("original_confirmation_spec_commitment")!=
                registry()["confirmation_full_spec_sha256"]):
        raise AssertionError("original world registry or sealed confirmation moved")
    root=Path(__file__).resolve().parent
    for rel,original_sha in cfg["source_git_blob_sha1"].items():
        contents=(root/rel).read_bytes()
        got=hashlib.sha1(b"blob "+str(len(contents)).encode()+b"\0"+contents).hexdigest()
        if got!=original_sha:
            raise AssertionError("source code changed after sham preregistration: "+rel)
    cfg["lock_sha256"]=hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return cfg


def raw_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as src:
        for part in iter(lambda:src.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()


def serialized(data) -> bytes:
    return (json.dumps(data,sort_keys=True,separators=(",",":"))+"\n").encode()


def scrambled_rank(seed: int, visit: int, variant: int) -> list[int]:
    """Fixed world-seed, visit and variant rank, independent of scored labels."""
    if variant not in VARIANTS or visit not in range(12) or seed<0:
        raise ValueError("invalid hash rank seed/visit/variant")
    return sorted(range(128),key=lambda feature: hashlib.sha256(
        RANK_SALT+seed.to_bytes(8,"big")+bytes((visit,variant,feature))).digest())


@torch.no_grad()
def source_and_hash_donors(sae,mean,scale,a,source_rank,*,n,hash_rank):
    """Select equal-cardinality disjoint top and hash-ranked active codes."""
    left,_,source=active_feature_policy(sae,mean,scale,a,source_rank,n=n)
    z=sae.codes((a-mean)/scale)
    physical_decoder=sae.decoder.weight.detach().T*scale[None,:]
    rank_positions={key:i for i,key in enumerate(hash_rank)}
    controls=[]
    controls_tensor=torch.zeros_like(a)
    for i in range(len(a)):
        active=source["all_active_features"][i]
        top=source["top_source_ranked_active"][i]
        excluded=set(top)
        available=[f for f in active if f not in excluded]
        if len(available)<n or len(active)!=16:
            raise AssertionError("source-active TopK has insufficient disjoint controls")
        chosen=sorted(available,key=lambda f:rank_positions[f])[:n]
        controls_tensor[i]=z[i,chosen]@physical_decoder[chosen]
        controls.append(chosen)
    if left.shape!=controls_tensor.shape or not torch.isfinite(controls_tensor).all():
        raise AssertionError("invalid hash-ranked GELU decoder donor")
    routing={
        "source_top_active":source["top_source_ranked_active"],
        "hash_selected_disjoint_active":controls,
        "all_active_feature_indices":source["all_active_features"],
        "features_per_site":n,
        "control_strategy":"hash-rank, source-label-blind, excludes original top active",
    }
    return left,controls_tensor,routing


@torch.no_grad()
def pair_donors(model,acts,candidate,hubs,sae,mean,scale,ranks,seed,variant):
    """Build the same original hubs-only lesion geometry at every chosen visit."""
    chosen=by_visit(candidate["nodes"])
    hub_set=by_visit(hubs)
    w=model.block.mlp_down.weight.detach()
    donors={}
    gates={}
    physical={}
    routes={}
    for visit,units in sorted(chosen.items()):
        n=len(units)
        if n<1 or n>8 or units & hub_set.get(visit,set()):
            raise AssertionError("source visit conflicts with hubs or exceeds TopK budget")
        activation=acts[:,visit,:]
        source,control,route=source_and_hash_donors(
            sae,mean,scale,activation,ranks[visit],n=n,
            hash_rank=scrambled_rank(seed,visit,variant))
        a,b,receipt=attenuate_pair(source,control,w)
        donors[visit]={"hubs":torch.zeros_like(a),"native":a,"sham":b}
        gates[visit]=masked(torch.ones_like(activation),
                            hub_set.get(visit,set()))[0]
        physical[visit]=receipt
        routes[visit]=route
    return donors,gates,physical,routes


def load_frozen_source(root: Path,lock_path: Path):
    """Verify the exact old source identities before any altered intervention."""
    world=json.loads((root/"world_spec.json").read_text())
    index=world["world_index"]
    if index not in range(8,16) or world["cohort"]!="development":
        raise PermissionError("only previously exposed NEC-03H-v2 development worlds")
    if world!=worlds()[index]:
        raise AssertionError("world function/positions changed")
    source=json.loads((root/"context_source_manifest.json").read_text())
    catalog,reconstructed=context_catalog(world)
    if source!=reconstructed:
        raise AssertionError("source context split differs from registered assignment")
    old=json.loads((root/"root_summary.json").read_text())
    if (old["schema"]!=SOURCE_FAMILY or old["world_sha256"]!=world["world_sha256"]
        or old["v2_protocol_lock_sha256"]!=raw_sha(lock_path)
        or old["original_confirmation_worlds_evaluated"]!=0
        or old["reserved_context_groups_evaluated"]!=0
        or not old["full_registered_v2_recipe"]):
        raise AssertionError("old v2 source or sealed-cohort receipt altered")
    if old["original_NEC03G_world_registry_sha256"]!=registry()["registry_sha256"]:
        raise AssertionError("old world registry is not original registered population")
    past=json.loads((root/"independent_source_audit.json").read_text())
    if past["root_summary_sha256"]!=raw_sha(root/"root_summary.json"):
        raise AssertionError("original independent source audit no longer matches")
    report=json.loads((root/"composed.report.json").read_text())
    if report["candidate_status"]!="SOURCE_SELECTED":
        raise AssertionError("no source-selected composed candidate")
    frozen=json.loads((root/"composed.source_freeze.json").read_text())
    if raw_sha(root/"composed.source_freeze.json")!=report["source_freeze_sha256"]:
        raise AssertionError("source-only ranked-feature roster modified")
    if frozen["world_sha256"]!=world["world_sha256"]:
        raise AssertionError("frozen source policy from another model")
    native_data=json.loads((root/"composed.report.json").read_text())["train"]
    if raw_sha(root/native_data["checkpoint_name"])!=native_data["checkpoint_sha256"]:
        raise AssertionError("native checkpoint bytes changed")
    torch_state=torch.load(root/native_data["checkpoint_name"],map_location="cpu",
                           weights_only=True)
    model=Looper()
    model.load_state_dict(torch_state["model"],strict=True)
    model.eval()
    if (model_digest(model)!=native_data["final_model_sha256"]
        or model_digest(model)!=frozen["model_sha256"]):
        raise AssertionError("trunk weights differ from original model digest")
    sae_manifest=frozen["source_sae"]
    if raw_sha(root/"composed.sae.pt")!=sae_manifest["source_sae_sha256"]:
        raise AssertionError("original source-only TopK16 checkpoint modified")
    sae_state=torch.load(root/"composed.sae.pt",map_location="cpu",weights_only=True)
    if sae_state["model_sha256"]!=model_digest(model):
        raise AssertionError("original TopK16 checkpoint trained on wrong source")
    sae=TopKSAE()
    sae.load_state_dict(sae_state["state"],strict=True)
    sae.eval()
    original_rows={}
    path=root/report["raw_paired_context_file"]
    if raw_sha(path)!=report["raw_paired_context_sha256"]:
        raise AssertionError("old raw paired logits changed")
    for line in path.read_text().splitlines():
        rec=json.loads(line)
        if rec["pair"]==["topk","topk_alt"]:
            original_rows[rec["source_context_index"]]=rec
    if set(original_rows)!=set(range(48)):
        raise AssertionError("original source-ranked comparison incomplete")
    return {
        "model":model,"world":world,"source":source,"catalog":catalog,
        "sae":sae,"mean":sae_state["mean"],"scale":sae_state["scale"],
        "original_report":report,
        "frozen_source":frozen,"original_rows":original_rows,
        "candidate":frozen["source_graph_candidate"],
        "original_root":old,"model_digest":model_digest(model),
    }


def check_logits(actual,record,*,atol=2e-5):
    expected=torch.tensor(record,dtype=actual.dtype)
    if expected.shape!=actual.shape or not torch.allclose(
            actual.detach(),expected,rtol=1e-5,atol=atol):
        raise AssertionError("replay differs from exact original source-frozen model")


def run(root: Path,out: Path,lock_path: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    panel_lock_path=Path(__file__).with_name("NEC03IB_SHAM_LOCK.json")
    panel_lock=verify_panel_lock(panel_lock_path)
    source=load_frozen_source(root,lock_path)
    if (source["original_root"]["v2_protocol_lock_sha256"]!=
            panel_lock["old_v2_protocol_lock_sha256"]):
        raise AssertionError("old source V2 lock differs from precommitted control panel")
    model=source["model"]
    world=source["world"]
    catalog=source["catalog"]
    candidate=source["candidate"]
    frozen=source["frozen_source"]
    rank=frozen["source_ranking_per_visit"]
    seed=world["trunk_seed"]
    sae=source["sae"]
    mean,scale=source["mean"],source["scale"]
    out.mkdir(parents=True,exist_ok=True)
    errors={v:{"clean":0,"hubs":0,"source":0,"sham":0} for v in VARIANTS}
    phys={v:[] for v in VARIANTS}
    traces={v:[] for v in VARIANTS}
    old_source_replay_count=0
    for context in range(48):
        group=catalog["development"][context*16:(context+1)*16]
        bits=as_tensor(group)
        labels=target_bits(bits,world)
        with torch.no_grad():
            clean=model.logits(bits)
        act=clean_activations(model,bits)
        # First independently replay the original source-top vs bottom-active
        # experiment on the SAME frozen model and contexts.
        baseline=source["original_rows"][context]
        original,doses,hubs,choices=build_pair(
            model,act,candidate,
            frozen["native_sham_policy"],rank,sae,mean,scale,
            frozen["graph_hubs"],("topk","topk_alt"))
        old_hubs=graft_forward(model,bits,original,hubs,"hubs")
        old_top=graft_forward(model,bits,original,hubs,"native")
        old_bottom=graft_forward(model,bits,original,hubs,"sham")
        check_logits(clean,baseline["clean_logits"])
        check_logits(old_hubs,baseline["hubs_logits"])
        check_logits(old_top,baseline["left_logits"])
        check_logits(old_bottom,baseline["right_logits"])
        old_source_replay_count+=1
        for variant in VARIANTS:
            donors,gates,dose,route=pair_donors(
                model,act,candidate,frozen["graph_hubs"],sae,mean,scale,
                rank,seed,variant)
            hubs_only=graft_forward(model,bits,donors,gates,"hubs")
            top=graft_forward(model,bits,donors,gates,"native")
            alternative=graft_forward(model,bits,donors,gates,"sham")
            for k,logits in (("clean",clean),("hubs",hubs_only),
                             ("source",top),("sham",alternative)):
                errors[variant][k]+=int((logits.argmax(1)!=labels).sum())
            phys[variant].append(dose)
            context_positions=source["source"]["context_positions"]
            traces[variant].append({
                "schema":SCHEMA,"world_id":world["world_id"],
                "variant":variant,"context_index":context,
                "source_context_bits":[group[0][j] for j in context_positions],
                "focal_mask_order":list(range(16)),
                "labels":labels.tolist(),
                "clean_logits":clean.detach().double().tolist(),
                "hubs_logits":hubs_only.double().tolist(),
                "source_logits":top.double().tolist(),
                "control_logits":alternative.double().tolist(),
                "local_direct_doses":{str(k):r for k,r in dose.items()},
                "active_feature_indices":{str(k):r for k,r in route.items()},
            })
    if model_digest(model)!=source["model_digest"]:
        raise AssertionError("read-only donor replay changed frozen native checkpoint")
    summaries=[]
    outfiles=[]
    for variant in VARIANTS:
        blob=b"".join(serialized(r) for r in traces[variant])
        filename=f"hashrank_{variant}.raw_contexts.jsonl"
        (out/filename).write_bytes(blob)
        stats=pair_dose_stats(phys[variant])
        rec={
            "variant":variant,"label_blind_hash_rank_sha256":hashlib.sha256(
                RANK_SALT+seed.to_bytes(8,"big")+bytes([variant])).hexdigest(),
            "original_source_active_rank_excluded_from_control":True,
            "source_only_feature_ranking_preserved":True,
            "pairwise_clean_errors":errors[variant]["clean"],
            "hubs_only_errors":errors[variant]["hubs"],
            "source_ranked_errors":errors[variant]["source"],
            "hash_control_errors":errors[variant]["sham"],
            "net_source_ranked_correct_gain":
                errors[variant]["sham"]-errors[variant]["source"],
            "dose":stats,
            "raw_context_sha256":hashlib.sha256(blob).hexdigest(),
            "raw_filename":filename,"context_rows":len(traces[variant]),
        }
        summaries.append(rec)
        outfiles.append(filename)
    row={
        "schema":SCHEMA,
        "world_id":world["world_id"],"world_index":world["world_index"],
        "world_sha256":world["world_sha256"],
        "original_run_id":FROZEN_ORIGINAL_RUN_ID,
        "old_source_report_sha256":raw_sha(root/"composed.report.json"),
        "original_ITT_signed_effect":source["original_root"]["primary_effect"]["root_primary_effect"],
        "original_primary_eligible":source["original_root"]["primary_effect"]["planned_world_valid"],
        "original_source_site_kind":source["original_root"]["source_site_kinds"]["composed"],
        "original_source_candidate_nodes":candidate["nodes"],
        "native_model_sha256":source["model_digest"],
        "source_sae_checkpoint_sha256":frozen["source_sae"]["source_sae_sha256"],
        "source_freeze_sha256":source["original_report"]["source_freeze_sha256"],
        "v2_lock_sha256":raw_sha(lock_path),
        "control_panel_lock_sha256":panel_lock["lock_sha256"],
        "original_source_active_top_vs_bottom_replayed_exact_contexts":old_source_replay_count,
        "source_contexts_used_by_new_panel":48,
        "reserved_contexts_scored":0,
        "confirmation_worlds_trained_or_scored":0,
        "hash_scramble_variants":[0,1,2,3],
        "controls_are_active_and_disjoint_from_original_top":True,
        "original_model_and_source_ranks_immutable":True,
        "comparisons":summaries,
        "posthoc_exploratory_only":True,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"source_blind_control_panel.json").write_bytes(serialized(row))
    print(json.dumps({"world":world["world_id"],
                      "original_score_replays":old_source_replay_count,
                      "raw_rows":48*len(VARIANTS),
                      "matched_and_signal_sufficient":sum(
                          x["dose"]["candidate_physically_comparable_and_signal_sufficient"]
                          for x in summaries),
                      "source_gain":{str(v):summaries[v]["net_source_ranked_correct_gain"]
                                     for v in VARIANTS},
                      "confirmation_worlds_scored":0},sort_keys=True),flush=True)
    return row


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--lock",type=Path,
                    default=Path(__file__).with_name("NEC03H_V2_LOCK.json"))
    args=ap.parse_args(argv)
    run(args.root,args.out,args.lock)


if __name__=="__main__":
    main()
