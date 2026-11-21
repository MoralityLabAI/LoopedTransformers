"""NEC-03K source-training-only direction-matched active SAE sham feasibility.

Eight PREVIOUSLY OPENED NEC-03H-v2 Boolean worlds only. Reuses frozen
900-update native trunk, source-trained TopK16 SAE, source-native lesion,
and frozen source-attribution ranks, all verified by original byte hashes.

At each source-training input / selected visit, match a disjoint equal-N
active-code control to the source donor's *cosine with the hubs-only
clean-chosen margin gradient in the physical 32-D MLP-down output*.
Enumerate every allowed active-code combination. Exactly equalize donor
output L2 norm by attenuation only. No true labels or evaluation contexts
enter the candidate selector or this calibration.

The first 16 of 80 original source training quartets are the entire
calibration population. Fail closed on absent/weak/poorly matched controls.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from hashlib import sha256
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.neuron_communities.nec03ib_active_shams import load_frozen_source
from mechinterp.neuron_communities.nec03j_direction_transport import (
    traced_forward, hash_file, oriented_margin,
)
from mechinterp.neuron_communities.nec03g_worlds import context_catalog, registry
from mechinterp.neuron_communities.matched_dose import by_visit, masked
from mechinterp.neuron_communities.nec03e_pairs import attenuate_pair, pair_dose_stats
from mechinterp.neuron_communities.topk_sae import active_feature_policy
from mechinterp.neuron_communities.source_sae import clean_activations
from research.pef07.pef07 import as_tensor, model_digest

SCHEMA = "NEC-03K/source-only-direction-matched-active-sham/v1"
SOURCE_CONTEXTS = 16
ANGLE_LIMIT = 0.25
MIN_ANGLE_COVERAGE = 0.80
MIN_PHYSICAL_COVERAGE = 0.80
MIN_MEDIAN_RETAINED = 0.25
SEED_SALT = b"NEC03K-v1-angle-matching-exhaustive-source-only"
SOURCE_WORLD_INDICES = tuple(range(8,16))


def canonical(obj) -> bytes:
    return (json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def verify_k_lock() -> dict:
    root=Path(__file__).resolve().parent
    lockfile=root/"NEC03K_SOURCE_LOCK.json"
    lock=json.loads(lockfile.read_text())
    if (lock.get("schema")!="NEC-03K/source-calibration-lock/v1"
        or lock.get("source_worlds")!=list(SOURCE_WORLD_INDICES)
        or lock.get("source_contexts")!=SOURCE_CONTEXTS
        or lock.get("max_absolute_cosine_gap")!=ANGLE_LIMIT
        or lock.get("min_angle_matched_fraction")!=MIN_ANGLE_COVERAGE
        or lock.get("min_physical_positive_fraction")!=MIN_PHYSICAL_COVERAGE
        or lock.get("min_median_retained_fraction")!=MIN_MEDIAN_RETAINED
        or lock.get("source_algorithm_salt")!=SEED_SALT.decode()
        or lock.get("original_v2_run_id")!=44878411694
        or lock.get("original_registry_sha256")!=registry()["registry_sha256"]
        or lock.get("original_confirmation_spec_sha256")!=
            registry()["confirmation_full_spec_sha256"]
        or lock.get("reserved_contexts_scored")!=0
        or lock.get("confirmation_worlds_scored")!=0):
        raise AssertionError("NEC03K source-only angle calibration lock invalid")
    for name,sha in lock["git_blob_sha1"].items():
        content=(root/name).read_bytes()
        blob=__import__("hashlib").sha1(
            b"blob "+str(len(content)).encode()+b"\x00"+content).hexdigest()
        if blob!=sha:
            raise AssertionError("source-calibrated geometry code changed: "+name)
    lock["lock_sha256"]=hash_file(lockfile)
    return lock


@torch.no_grad()
def exhaustive_direction_match(sae, mean, scale, activations, source_ranking,
                               gradient32, down_weight, *, count: int):
    """Return source and closest-angle *disjoint* active-code control.

    Search the complete subset family C(16-N,N), with no correctness data.
    Numerical closeness in native 32D MLP output angle is independent of
    the original signal norm; shared min-dose attenuation is separate.
    """
    if (activations.ndim!=2 or activations.shape[1]!=64
        or gradient32.shape!=(len(activations),32)
        or down_weight.shape!=(32,64) or count not in range(1,9)
        or len(source_ranking)!=128 or set(source_ranking)!=set(range(128))):
        raise ValueError("incompatible native GELU/directional matching coordinates")
    top_vector,_,routes=active_feature_policy(
        sae,mean,scale,activations,source_ranking,n=count)
    z=sae.codes((activations-mean)/scale)
    decoder64=sae.decoder.weight.detach().T*scale[None,:]
    decoder32=decoder64@down_weight.T
    source32=top_vector@down_weight.T
    control64=torch.zeros_like(top_vector)
    route_receipts=[]
    for i in range(len(activations)):
        active=routes["all_active_features"][i]
        source_ids=routes["top_source_ranked_active"][i]
        others=sorted(set(active)-set(source_ids))
        if len(active)!=16 or len(others)<count or len(source_ids)!=count:
            raise AssertionError("active feature support or equal component budget invalid")
        candidate_tuples=list(combinations(others,count))
        if not candidate_tuples:
            raise AssertionError("no disjoint active-code candidate")
        idx=torch.tensor(candidate_tuples,dtype=torch.long,device=z.device)
        # [C,N,32] -> [C,32], up to C(12,4)=495 per input.
        out32=(z[i,idx].unsqueeze(-1)*decoder32[idx]).sum(dim=1)
        source_norm=source32[i].norm()
        gradnorm=gradient32[i].norm()
        candidate_norms=out32.norm(dim=-1)
        eps=1e-10
        src_cos=float((source32[i]*gradient32[i]).sum()/
                      (source_norm*gradnorm).clamp_min(eps))
        controls_cos=(out32@gradient32[i])/(
            candidate_norms*gradnorm).clamp_min(eps)
        error=(controls_cos-src_cos).abs()
        index=int(torch.argmin(error))
        selected=list(candidate_tuples[index])
        control64[i]=z[i,selected]@decoder64[selected]
        physical_source=float(source_norm)
        physical_control=float(candidate_norms[index])
        gap=float(error[index])
        grad_is_valid=bool(float(gradnorm)>1e-7)
        cos_match=bool(gap<=ANGLE_LIMIT and grad_is_valid
                       and physical_source>1e-6 and physical_control>1e-6)
        route_receipts.append({
            "source_top_active":source_ids,
            "all_active":active,
            "selected_disjoint_control":selected,
            "candidate_count":len(candidate_tuples),
            "physical_source_norm":physical_source,
            "physical_control_norm":physical_control,
            "source_cosine":src_cos,
            "control_cosine":float(controls_cos[index]),
            "absolute_cosine_gap":gap,
            "gradient_norm":float(gradnorm),
            "within_angle_gate":cos_match,
            "source_projection32":source32[i].double().tolist(),
            "hub_gradient32":gradient32[i].double().tolist(),
            # Audit can independently reconstruct every candidate direction.
            "active_output_contributions32": [
                {"id":int(f),"vector":(z[i,f]*decoder32[f]).double().tolist()}
                for f in active],
        })
    return top_vector, control64, route_receipts


def source_pilot(root: Path, output: Path, original_v2_lock: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock=verify_k_lock()
    prepared=load_frozen_source(root,original_v2_lock)
    model=prepared["model"]
    w=prepared["world"]
    if w["world_index"] not in SOURCE_WORLD_INDICES:
        raise PermissionError("NEC03K cannot open untouched development or confirmation worlds")
    catalog=prepared["catalog"]
    candidate=prepared["candidate"]
    frozen=prepared["frozen_source"]
    model_sha=prepared["model_digest"]
    selected=by_visit(candidate["nodes"])
    hubs=by_visit(frozen["graph_hubs"])
    rank=frozen["source_ranking_per_visit"]
    sae=prepared["sae"]
    mean,scale=prepared["mean"],prepared["scale"]
    down=model.block.mlp_down.weight.detach()
    output.mkdir(parents=True,exist_ok=True)
    raw=[]
    all_doses=[]
    coverage=[]
    for ctx in range(SOURCE_CONTEXTS):
        # Source-only input membership; absolutely no development/reserved read.
        bits=as_tensor(catalog["train"][ctx*16:(ctx+1)*16])
        clean=traced_forward(model,bits,mode="clean")
        chosen=clean["logits"].argmax(1)
        acts=clean_activations(model,bits)
        baseline_donors={}
        gates={}
        for visit in selected:
            baseline_donors[visit]={"hubs":torch.zeros(16,64)}
            gate=torch.zeros(64)
            if visit in hubs:
                gate[list(sorted(hubs[visit]))]=1.
            gates[visit]=gate
        hubs_trace=traced_forward(
            model,bits,mode="hubs",orientation=chosen,
            donors=baseline_donors,gates=gates,derivative=True)
        site_data={}
        dose_context={}
        for visit,units in sorted(selected.items()):
            top,control,route=exhaustive_direction_match(
                sae,mean,scale,acts[:,visit,:],rank[visit],
                hubs_trace["grad_down"][visit],down,count=len(units))
            left,right,doses=attenuate_pair(top,control,down)
            # No task label, outcome or novel causal graft used to choose controls.
            dose_context[visit]=doses
            eligible_per_site=[
                bool(r["within_angle_gate"] and doses["shared_nonzero"][i]
                     and doses["matched_within_1pct"][i])
                for i,r in enumerate(route)]
            coverage.extend(eligible_per_site)
            site_data[str(visit)]={
                "routes":route,
                "dose":doses,
                "angle_passed_and_positive_with_matched_dose":eligible_per_site,
                "source_norm_after_matching":(
                    left@down.T).norm(dim=1).double().tolist(),
                "control_norm_after_matching":(
                    right@down.T).norm(dim=1).double().tolist(),
            }
        all_doses.append(dose_context)
        raw.append({
            "schema":SCHEMA,"world_id":w["world_id"],
            "source_context_index":ctx,
            "source_context_bits":[int(x) for x in
                prepared["source"]["contexts_ordered"][ctx]],
            "input_focal_masks":list(range(16)),
            "clean_chosen_class":chosen.tolist(),
            "clean_logits":clean["logits"].double().tolist(),
            "hubs_logits":hubs_trace["logits"].double().tolist(),
            "sites":site_data,
            "output_labels_consulted_for_selection":False,
            "source_training_context_only":True,
        })
    signal=pair_dose_stats(all_doses)
    angle_and_dose_rate=sum(coverage)/len(coverage)
    angle_rate=sum(r["within_angle_gate"]
        for record in raw for site in record["sites"].values()
        for r in site["routes"])/len(coverage)
    summary={
        "schema":SCHEMA,"world_id":w["world_id"],"world_index":w["world_index"],
        "world_sha256":w["world_sha256"],
        "original_model_sha256":model_sha,
        "original_source_rank_freeze_sha256":hash_file(
            root/"composed.source_freeze.json"),
        "original_native_checkpoint_sha256":prepared["original_report"]["train"]["checkpoint_sha256"],
        "original_topk16_checkpoint_sha256":frozen["source_sae"]["source_sae_sha256"],
        "original_v2_lock_sha256":hash_file(original_v2_lock),
        "NEC03K_source_lock_sha256":lock["lock_sha256"],
        "candidate_source_site_kind":prepared["original_root"]["source_site_kinds"]["composed"],
        "source_training_context_quartets":SOURCE_CONTEXTS,
        "source_focal_histories":SOURCE_CONTEXTS*16,
        "original_hubs_gradient_coordinate":"post-mlp_down 32D clean-chosen margin, hubs-only trajectory",
        "angle_limit":ANGLE_LIMIT,
        "angle_matched_fraction":angle_rate,
        "angle_and_physical_matched_fraction":angle_and_dose_rate,
        "dose":signal,
        "source_null_feasible":(
            angle_and_dose_rate>=MIN_ANGLE_COVERAGE
            and signal["candidate_physically_comparable_and_signal_sufficient"]),
        "raw_context_file":"nec03k_source_geometry.jsonl",
        "source_selection_uses_task_labels":False,
        "all_original_ITT_and_confirm_outcomes_unchanged":True,
        "heldout_development_contexts_scored":0,
        "reserved_contexts_scored":0,
        "confirmation_worlds_scored":0,
        "inference_status":"source-only feasibility, no heldout causal effect",
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    blob=b"".join(canonical(item) for item in raw)
    (output/summary["raw_context_file"]).write_bytes(blob)
    summary["raw_context_sha256"]=sha256(blob).hexdigest()
    (output/"nec03k_source_calibration.json").write_bytes(canonical(summary))
    if model_digest(model)!=model_sha:
        raise AssertionError("source-only direction matching mutated the native model")
    print("NEC03K_FROZEN_SOURCE_GEOMETRY_SAVED")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--original-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--old-lock",type=Path,default=Path(__file__).with_name("NEC03H_V2_LOCK.json"))
    args=p.parse_args(argv)
    source_pilot(args.original_root,args.out,args.old_lock)


if __name__=="__main__":
    main()
