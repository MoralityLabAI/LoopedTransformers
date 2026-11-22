"""NEC-03X frozen source-only continuous physical-subspace necessity.

Original eight native Boolean development worlds, both source-trained
teachers, immutable NEC03O native/SAE checkpoints, original training-
source groups 40..47 only. Source coherent quartet from NEC03V frozen.
New disjoint sham uses 495 supports with 4 box-constrained nonnegative
decoder weights, deterministic 32-step physical 32D optimization.
"""
from __future__ import annotations

import argparse
from hashlib import sha1,sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import median

import torch

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,labels,encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
from mechinterp.neuron_communities.nec03v_coherent_source import (
    source_coherent_group,physical_cohesion,original_case_signed_margin,
)
from mechinterp.neuron_communities.nec03x_solver import (
    STEPS,COEFF_MULTIPLIER,RESTARTS,ANGLE_GATE,COHESION_GATE,
    CROSS_GRAM_FLOOR,OUTPUT_VECTOR_COS_FLOOR,
    SHARED_DOSE_FLOOR,DOSE_RELATIVE_TOL,COHESION_WEIGHT,
    DIRECTION_WEIGHT,select_continuous_sham,equalize_removal,cosine,
)
from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.nec03m_source_pilot import sha_file
from research.pef07.pef07 import as_tensor,model_digest

SCHEMA="NEC-03X/source-only-continuous-disjoint-physical-null/v1"
LOCK_NAME="NEC03X_SOURCE_LOCK.json"
ROOTS=tuple(range(8))
PANEL=tuple(range(40,48))
ELIGIBLE=(0,1,2,4,6,7)
ITT_ZERO=(3,5)
FOCAL=32
CASES=256
K=4
PRIMARY_FLOOR=.02
MODES=(
    "clean","hubs","native_nonhub","full_affine","contrib16_only",
    "continuous4_source","continuous4_sham",
    "discrete4_source","discrete4_sham",
)


def read(p):
    return json.loads(Path(p).read_text())


def verify_lock():
    cfg=read(Path(__file__).with_name(LOCK_NAME))
    exact={
        "schema":"NEC-03X/frozen-source-continuous-physical-sham/v1",
        "original_root_indices":list(ROOTS),
        "original_source_group_indices":list(PANEL),
        "full_focal_masks":FOCAL,
        "source_cases_per_arm":CASES,
        "source_eligible_root_indices":list(ELIGIBLE),
        "prior_source_ITT_zero_root_indices":list(ITT_ZERO),
        "original_native_A_run":44898361742,
        "original_sae_A_run":44899750508,
        "original_native_sae_B_run":44900628360,
        "five_input_registry_sha256":registration()["registry_sha256"],
        "four_input_confirmation_sha256":old_registry()["confirmation_full_spec_sha256"],
        "sae_positive_codes":16,
        "source_coherent_group_size":K,
        "source_selector":"original highest source-ranked active seed + 3 nearest cosine-aligned physical 32D decoder columns",
        "sham_supports_enumerated":495,
        "sham_continuous_coefficients":4,
        "projected_solver_steps":STEPS,
        "projected_solver_restarts":RESTARTS,
        "coefficient_upper_multiplier":COEFF_MULTIPLIER,
        "optimizer_loss":"sum_{j=0}^31 (sum_i coeff_i * physical_32d_column_i[j] - source_32d[j])^2/2",
        "optimizer_step":"projected gradient, step 1/max(sum_i ||physical_32d_column_i||^2,1e-10), box [0,3*original_active_code]",
        "control_selection":"among feasible supports minimize angle_gap + 0.25*cohesion_gap + 0.25*(1-output_vector_cosine); otherwise keep global objective minimum and mark unavailable",
        "gradient_angle_gap_max":ANGLE_GATE,
        "physical_group_cohesion_gap_max":COHESION_GATE,
        "cross_gram_absolute_overlap_min":CROSS_GRAM_FLOOR,
        "source_control_output_cosine_min":OUTPUT_VECTOR_COS_FLOOR,
        "minimum_shared_32d_dose":SHARED_DOSE_FLOOR,
        "maximum_matched_32d_dose_relative_error":DOSE_RELATIVE_TOL,
        "all_modes":list(MODES),
        "primary_effect":"((composed continuous_sham_correct - continuous_source_correct)-(additive continuous_sham_correct - continuous_source_correct))/256, roots3/5 fixed zero, eight-root mean",
        "primary_comparison_count":1,
        "primary_meaningful_floor":PRIMARY_FLOOR,
        "qualified_root_positive_gate":5,
        "matched_input_gate_per_eligible_arm":205,
        "root_uncertainty_exploratory_signflip_p_gate":.05,
        "scientific_inference":"eight-root original-source mechanism pilot only, no population significance or 59-root causal-sham calibration",
        "secondary_margin_norm":"max(1.0, median_abs_true_label_full_affine_margin on 256 original-source cases within native model arm)",
        "original_discrete_sham_secondary":"same original 495-disjoint activation quartet with fixed angle+0.5*cohesion joint objective and original angle<=.25, cohesion<=.50",
        "zero_effect_unavailable":"both source/control donor vectors equal unchanged full-affine at every selected native visit",
        "new_native_or_SAE_training_updates":0,
        "new_development_eval_groups_scored":0,
        "reserved_groups_scored":0,
        "new_recruited_worlds_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(cfg.get(k)!=v for k,v in exact.items()):
        raise AssertionError("NEC03X source-only model population or solver changed")
    root=Path(__file__).resolve().parents[2]
    for filename,pin in cfg["git_blob_sha1"].items():
        blob=(root/filename).read_bytes()
        digest=sha1(b"blob "+str(len(blob)).encode()+b"\x00"+blob).hexdigest()
        if digest!=pin:raise AssertionError("NEC03X preregistered source blob changed: "+filename)
    return cfg,sha_file(Path(__file__).with_name(LOCK_NAME))


def group_measure(prepared,group,histories):
    if group not in PANEL or len(histories)!=FOCAL:
        raise PermissionError("only original complete training-source groups 40..47")
    world=prepared["world"]
    arm=prepared["arm"]
    model=prepared["model"]
    x=as_tensor(histories)
    clean=traced_forward(model,x,mode="clean")
    orient=clean["logits"].argmax(1)
    sites=prepared["source_sites"]
    hubs=prepared["source_hubs"]
    gates={}
    for visit in sites:
        mask=torch.zeros(64)
        if visit in hubs:mask[sorted(hubs[visit])]=1.
        gates[visit]=mask
    hubs_donors={visit:{"hubs":torch.zeros(FOCAL,64)} for visit in sites}
    lesion=traced_forward(model,x,mode="hubs",orientation=orient,
                          donors=hubs_donors,gates=gates)
    variants={k:{} for k in MODES if k not in ("clean","hubs")}
    good_cont=torch.full((FOCAL,),bool(sites),dtype=torch.bool)
    good_discrete=torch.full((FOCAL,),bool(sites),dtype=torch.bool)
    if sites:
        acts=clean_activations(model,x)
    else:
        acts=None
    down=model.block.mlp_down.weight.detach()
    full={}
    code={}
    columns={}
    site_records={}
    for visit,nodes in sorted(sites.items()):
        nonhub=1.-gates[visit]
        native=acts[:,visit,:]
        ae=prepared["sae"]
        with torch.no_grad():
            z=ae.codes((native-prepared["mean"])/prepared["scale"])
            physcols=ae.decoder.weight.detach().T*prepared["scale"][None,:]
            affine=ae.decoder(z)*prepared["scale"]+prepared["mean"]
            feature16=z@physcols
            offset=prepared["mean"]+ae.decoder.bias.detach()*prepared["scale"]
        if not torch.allclose(affine,feature16+offset,atol=2e-5,rtol=1e-5):
            raise AssertionError("exact original full affine SAE decoder changed")
        full[visit]=affine*nonhub
        code[visit]=z
        columns[visit]=physcols
        variants["native_nonhub"][visit]=native*nonhub
        variants["full_affine"][visit]=full[visit]
        variants["contrib16_only"][visit]=feature16*nonhub
    if sites:
        base=traced_forward(model,x,mode="native",orientation=orient,
            donors={visit:{"native":vec} for visit,vec in full.items()},
            gates=gates,derivative=True)
    else:
        base=clean
    for visit,nodes in sorted(sites.items()):
        nonhub=1.-gates[visit]
        z=code[visit]
        decoder=columns[visit]
        phys=(decoder*nonhub[None,:])@down.T
        norms=phys.norm(dim=1)
        dot=(phys@phys.T)/(norms[:,None]*norms[None,:]).clamp_min(1e-10)
        rank={feature:i for i,feature in enumerate(prepared["ranks"][visit])}
        grad=base["grad_down"][visit]
        src=torch.zeros(FOCAL,64)
        ctrl=torch.zeros(FOCAL,64)
        disc_src=torch.zeros(FOCAL,64)
        disc_ctrl=torch.zeros(FOCAL,64)
        case_meta=[]
        for mask in range(FOCAL):
            active=torch.where(z[mask]>1e-7)[0].tolist()
            record={"active_ids":active,"full_topk_support":len(active)==16,
                    "source_ids":[],"continuous":None,"discrete":None}
            if len(active)!=16:
                good_cont[mask]=False
                good_discrete[mask]=False
                case_meta.append(record)
                continue
            src_ids=source_coherent_group(active,rank,dot,K)
            record["source_ids"]=src_ids
            selected,coefs,detail=select_continuous_sham(
                active,src_ids,z[mask],phys,dot,grad[mask])
            source64=(z[mask,src_ids]@decoder[src_ids])*nonhub
            sham64=(coefs@decoder[selected])*nonhub
            src[mask]=source64
            ctrl[mask]=sham64
            record["continuous"]=detail
            if not detail["selected_candidate_quality_valid"]:
                good_cont[mask]=False
            # Freeze the NEC03V discrete disjoint group selection as a
            # secondary new-source-window descriptive comparison.
            choices=list(combinations(sorted(set(active)-set(src_ids)),4))
            ids=torch.tensor(choices,dtype=torch.long)
            control32=(z[mask,ids].unsqueeze(-1)*phys[ids]).sum(1)
            src32=source64@down.T
            gap=(cosine(control32,grad[mask])-cosine(src32,grad[mask])).abs()
            cohesion_src=physical_cohesion(src_ids,dot)
            cohesion_cand=torch.zeros(len(choices))
            for a in range(4):
                for b in range(a+1,4):
                    cohesion_cand+=dot[ids[:,a],ids[:,b]]
            cohesion_cand/=6
            gap_coh=(cohesion_cand-cohesion_src).abs()
            scores=gap+.5*gap_coh
            j=int(torch.argmin(scores))
            prior_ids=list(choices[j])
            disc_src[mask]=source64
            disc_ctrl[mask]=(z[mask,prior_ids]@decoder[prior_ids])*nonhub
            record["discrete"]={
                "sham_ids":prior_ids,
                "angle_gap":float(gap[j]),"cohesion_gap":float(gap_coh[j]),
                "joint_score":float(scores[j]),
                "original_candidates":495,
                "valid_physical_match":bool(
                    float(gap[j])<=.25 and float(gap_coh[j])<=.50
                    and grad[mask].norm()>1e-7),
            }
            if not record["discrete"]["valid_physical_match"]:
                good_discrete[mask]=False
            case_meta.append(record)
        cont,dose_cont=equalize_removal(src,ctrl,down)
        disc,dose_discrete=equalize_removal(disc_src,disc_ctrl,down)
        for m in range(FOCAL):
            good_cont[m]&=bool(dose_cont["dose_match_within_1pct"][m] and
                               dose_cont["positive_matched_dose"][m])
            good_discrete[m]&=bool(dose_discrete["dose_match_within_1pct"][m] and
                                   dose_discrete["positive_matched_dose"][m])
        variants["continuous4_source"][visit]=full[visit]-cont["source"]
        variants["continuous4_sham"][visit]=full[visit]-cont["sham"]
        variants["discrete4_source"][visit]=full[visit]-disc["source"]
        variants["discrete4_sham"][visit]=full[visit]-disc["sham"]
        site_records[str(visit)]={
            "native_selected_units":sorted(nodes),
            "graph_hubs":sorted(hubs.get(visit,[])),
            "full_affine_32d_output_dose":(full[visit]@down.T).norm(dim=1).tolist(),
            "full_affine_32d_reconstruction_error":(
                ((full[visit]-variants["native_nonhub"][visit])@down.T).norm(dim=1)/
                (variants["native_nonhub"][visit]@down.T).norm(dim=1).clamp_min(1e-6)
            ).tolist(),
            "case_geometry":case_meta,
            "continuous_pair_dose":dose_cont,
            "discrete_pair_dose":dose_discrete,
        }
    for visit in sites:
        for availability,prefix in ((good_cont,"continuous4"),(good_discrete,"discrete4")):
            for side in ("source","sham"):
                name=f"{prefix}_{side}"
                variants[name][visit]=full[visit]+(
                    variants[name][visit]-full[visit])*availability[:,None]
    trace={"clean":clean,"hubs":lesion}
    for name,vecs in variants.items():
        donors={visit:{"native":v} for visit,v in vecs.items()}
        with torch.no_grad():
            trace[name]=traced_forward(model,x,mode="native",
                orientation=orient,donors=donors,gates=gates)
    for gate,family in ((good_cont,"continuous4"),(good_discrete,"discrete4")):
        bad=~gate
        if not torch.allclose(trace[f"{family}_source"]["logits"][bad],
                              trace[f"{family}_sham"]["logits"][bad],atol=1e-6):
            raise AssertionError("invalid input produced a nonzero active sham contrast")
    return {
        "schema":SCHEMA,"group":group,"source_input_histories":histories,
        "focal_masks":list(range(32)),
        "clean_orientation":orient.tolist(),
        "teacher_labels_used_for_sham_selection":False,
        "site_geometry":site_records,
        "continuous_available_all_visits":good_cont.tolist(),
        "discrete_available_all_visits":good_discrete.tolist(),
        "logits":{k:v["logits"].tolist() for k,v in trace.items()},
        "mean_recurrent_memory_drift_from_full_affine":{
            k:(v["memory"]-base["memory"]).norm(dim=-1).mean(dim=0).tolist()
            for k,v in trace.items()},
    }


def run(index,source,out,*,reset=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg,lock_sha=verify_lock()
    if index not in ROOTS:
        raise PermissionError("only eight previously source-trained original worlds")
    world=worlds()[index]
    out.mkdir(parents=True,exist_ok=True)
    arms={}
    for arm in ("composed","additive"):
        prepared=load_source_arm(index,arm,source,reset=reset)
        prepared["arm"]=arm
        if prepared["world"]["world_sha256"]!=world["world_sha256"]:
            raise AssertionError("source model native root changed")
        counts={m:0 for m in MODES}
        valid_cont=valid_disc=0
        raw=[]
        margin={m:[] for m in ("full_affine","continuous4_source","continuous4_sham")}
        for group in PANEL:
            histories=prepared["catalog"]["train"][group*FOCAL:(group+1)*FOCAL]
            record=group_measure(prepared,group,histories)
            y=labels(as_tensor(histories),world,control=arm=="additive").tolist()
            pred={}
            for m in MODES:
                scores=record["logits"][m]
                classes=[int(p[1]>p[0]) for p in scores]
                pred[m]=classes
                counts[m]+=sum(int(a==b) for a,b in zip(classes,y))
                if m in margin:
                    margin[m].extend(original_case_signed_margin(scores,y))
            valid_cont+=sum(record["continuous_available_all_visits"])
            valid_disc+=sum(record["discrete_available_all_visits"])
            record["arm"]=arm
            record["original_source_teacher_labels"]=y
            record["original_predictions"]=pred
            raw.append(record)
        rawbytes=b"".join(encoded(row) for row in raw)
        path=out/f"{arm}.nec03x_continuous_source.jsonl"
        path.write_bytes(rawbytes)
        scale=max(1.,median(abs(x) for x in margin["full_affine"]))
        delta=sum((b-a)/scale for a,b in zip(
            margin["continuous4_source"],margin["continuous4_sham"]))/CASES
        arms[arm]={
            "source_model_sha256":prepared["model_sha256"],
            "source_sae_sha256":prepared["sae_sha256"],
            "source_rank_sha256":prepared["rank_sha256"],
            "original_source_receipt_sha256":prepared["source_report_sha256"],
            "original_source_eligible":prepared["original_arm_source_eligible"],
            "source_candidate_present":bool(prepared["source_candidate"]),
            "source_group_indices":list(PANEL),"source_cases":CASES,
            "correct_counts":counts,
            "matched_continuous_inputs":valid_cont,
            "matched_discrete_inputs":valid_disc,
            "continuous_sham_minus_source_correct":
                counts["continuous4_sham"]-counts["continuous4_source"],
            "discrete_sham_minus_source_correct":
                counts["discrete4_sham"]-counts["discrete4_source"],
            "secondary_within_model_margin_scale":scale,
            "secondary_normalized_continuous_sham_minus_source_margin":float(delta),
            "raw_file":path.name,"raw_sha256":sha256(rawbytes).hexdigest(),
        }
        if model_digest(prepared["model"])!=prepared["model_sha256"]:
            raise AssertionError("continuous source experiment changed native weights")
    effect=(arms["composed"]["continuous_sham_minus_source_correct"]-
            arms["additive"]["continuous_sham_minus_source_correct"])/CASES
    result={
        "schema":SCHEMA,
        "world_index":index,"world_id":world["world_id"],
        "world_sha256":world["world_sha256"],
        "source_manifest_sha256":context_catalog(world)[1]["source_sha256"],
        "five_input_registry_sha256":registration()["registry_sha256"],
        "four_input_confirmation_sha256":old_registry()["confirmation_full_spec_sha256"],
        "source_lock_sha256":lock_sha,
        "source_training_groups_scored":list(PANEL),
        "source_cases_per_model_arm":CASES,
        "original_paired_source_eligible":index in ELIGIBLE,
        "original_ITT_zero":index in ITT_ZERO,
        "raw_continuous_primary_root_DID":effect,
        "primary_original_ITT_root_DID":effect if index in ELIGIBLE else 0.,
        "arms":arms,
        "model_or_SAE_optimizer_updates":0,
        "new_development_eval_groups_scored":0,
        "reserved_groups_scored":0,"new_recruited_worlds_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"nec03x_source_root_report.json").write_bytes(encoded(result))
    print("NEC03X_FROZEN_CONTINUOUS_SOURCE_ROOT_WRITTEN",world["world_id"])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--reset-root",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.world_index,a.source_root,a.out,reset=a.reset_root)
