"""NEC-03Z-CAL: original-source-only sham functional-inertness dose calibration.

Only the previously trained dev5-000..007 Boolean model world roots,
original source training groups 48..55. Five fixed physical common-dose
fractions on the existing NEC03X label-blind sham selection and matched
source quartet. The source/sham family, graph, model and SAE never train.

56..63 is a SEALED candidate analysis panel: never imported or scored.
"""
from __future__ import annotations

import argparse
from hashlib import sha1, sha256
import json
from pathlib import Path
from statistics import median
import torch

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds, registration, context_catalog, labels, encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
from mechinterp.neuron_communities.nec03v_coherent_source import source_coherent_group
from mechinterp.neuron_communities.nec03x_solver import (
    select_continuous_sham, equalize_removal,
)
from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.nec03m_source_pilot import sha_file
from research.pef07.pef07 import as_tensor, model_digest

SCHEMA = "NEC-03Z/source-only-inertness-dose-CAL/v1"
LOCK = "NEC03Z_CAL_LOCK.json"
ROOTS = tuple(range(8))
ELIGIBLE = (0,1,2,4,6,7)
ITT_ZERO = (3,5)
CAL = tuple(range(48,56))
SEALED = tuple(range(56,64))
DOSES = (1.0, 0.5, 0.25, 0.125, 0.0625)
TAGS = ("1000","0500","0250","0125","00625")
CASES = 256
MASKS = 32
MAX_DAMAGE = 0.02
MIN_MATCHED = 205
MIN_RELATIVE_DOSE = 0.05
MIN_BAND_ACCURACY = .60
MAX_BAND_ACCURACY = .95
MIN_UNSATURATED_ROOTS_PER_FAMILY = 2
MODES = ("clean","hubs","native_nonhub","full_affine") + tuple(
    f"{side}_d{tag}" for tag in TAGS for side in ("source","sham")
)
NAME_TO_DOSE = dict(zip(TAGS,DOSES))


def verify_lock():
    path = Path(__file__).with_name(LOCK)
    v = json.loads(path.read_text())
    fixed = {
        "schema":"NEC-03Z/frozen-original-source-sham-inertness-CAL/v1",
        "roots":list(ROOTS),
        "paired_eligible_roots":list(ELIGIBLE),
        "original_ITT_zero_roots":list(ITT_ZERO),
        "calibration_source_groups":list(CAL),
        "sealed_source_analysis_groups":list(SEALED),
        "full_masks_per_group":MASKS,
        "old_train_cases_per_arm":CASES,
        "old_native_A_run":44898361742,
        "old_SAE_A_run":44899750508,
        "old_native_SAE_B_run":44900628360,
        "five_input_registry_sha256":registration()["registry_sha256"],
        "old_four_input_confirmation_sha256":old_registry()["confirmation_full_spec_sha256"],
        "locked_continuous_solver_family":"NEC03X 495-disjoint-quartet 32-step projected box coefficient solver, unchanged",
        "dose_fractions":list(DOSES),
        "dose_tags":list(TAGS),
        "scored_modes":list(MODES),
        "damage_ceiling_per_original_qualified_arm":MAX_DAMAGE,
        "minimum_valid_controls_per_original_qualified_arm":MIN_MATCHED,
        "median_min_site_matched_removal_to_source_norm_floor":MIN_RELATIVE_DOSE,
        "baseline_accuracy_headroom_band":[MIN_BAND_ACCURACY,MAX_BAND_ACCURACY],
        "minimum_eligible_headroom_roots_per_teacher_family":MIN_UNSATURATED_ROOTS_PER_FAMILY,
        "selection_rule":"highest dose satisfying per-arm sham c2w<=2pct among baseline-correct physical matches, >=205/256 matched per eligible arm, median per-case minimum matched-dose/source-natural-dose >=0.05 per eligible arm, and >=2 eligible roots in 60-95pct full-affine accuracy per teacher family",
        "failure_policy":"if no dose satisfies all conditions, SEAL groups56..63 without model inference; no efficacy evaluation",
        "no_label_based_sham_routing":True,
        "original_sealed_group_scoring":0,
        "original_development_eval_scoring":0,
        "reserved_scoring":0,
        "recruited_world_scoring":0,
        "embargoed_five_input_scoring":0,
        "old_four_input_confirmation_scoring":0,
        "model_SAE_optimizer_updates":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(v.get(k)!=vv for k,vv in fixed.items()):
        raise AssertionError("NEC03Z original source-only calibration method changed")
    repo=Path(__file__).resolve().parents[2]
    for f,pin in v["git_blob_sha1"].items():
        b=(repo/f).read_bytes()
        blob=sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
        if blob!=pin:raise AssertionError("NEC03Z frozen source blob changed "+f)
    return v,sha_file(path)


def physical_median(values):
    return float(median(values)) if values else None


def group_measure(prepared, group, histories):
    if group not in CAL or len(histories)!=MASKS:
        raise PermissionError("NEC03Z may score only original training groups 48..55")
    model=prepared["model"]
    x=as_tensor(histories)
    clean=traced_forward(model,x,mode="clean")
    orientation=clean["logits"].argmax(1)
    sites=prepared["source_sites"]
    hubs=prepared["source_hubs"]
    gates={}
    for visit in sites:
        gate=torch.zeros(64)
        if visit in hubs:gate[sorted(hubs[visit])]=1
        gates[visit]=gate
    lesion=traced_forward(
        model,x,mode="hubs",orientation=orientation,
        donors={v:{"hubs":torch.zeros(MASKS,64)} for v in sites},gates=gates
    )
    variants={mode:{} for mode in MODES if mode not in ("clean","hubs")}
    full={}
    code={}
    columns={}
    acts=clean_activations(model,x) if sites else None
    down=model.block.mlp_down.weight.detach()
    for visit in sites:
        native=acts[:,visit,:]
        nonhub=1-gates[visit]
        sae=prepared["sae"]
        with torch.no_grad():
            z=sae.codes((native-prepared["mean"])/prepared["scale"])
            dec=sae.decoder.weight.detach().T*prepared["scale"][None,:]
            aff=sae.decoder(z)*prepared["scale"]+prepared["mean"]
            offset=prepared["mean"]+sae.decoder.bias.detach()*prepared["scale"]
        if not torch.allclose(aff,z@dec+offset,atol=2e-5,rtol=1e-5):
            raise AssertionError("original frozen source full-affine identity changed")
        full[visit]=aff*nonhub
        code[visit]=z
        columns[visit]=dec
        variants["native_nonhub"][visit]=native*nonhub
        variants["full_affine"][visit]=full[visit]
    reference=(
        traced_forward(
            model,x,mode="native",orientation=orientation,
            donors={v:{"native":t} for v,t in full.items()},
            gates=gates,derivative=True
        ) if sites else clean
    )
    valid_base=torch.full((MASKS,),bool(sites),dtype=torch.bool)
    # Per-case worst recurrent visit relative dose remains full denominator.
    min_relative=torch.full((MASKS,),float("inf"))
    geometry={}
    for visit,nodes in sorted(sites.items()):
        nonhub=1-gates[visit]
        z=code[visit]
        dec=columns[visit]
        phys=(dec*nonhub[None,:])@down.T
        norms=phys.norm(dim=1)
        cosines=phys@phys.T/(norms[:,None]*norms[None,:]).clamp_min(1e-10)
        ranking={f:i for i,f in enumerate(prepared["ranks"][visit])}
        source=torch.zeros(MASKS,64)
        sham=torch.zeros(MASKS,64)
        case_meta=[]
        for mask in range(MASKS):
            active=torch.where(z[mask]>1e-7)[0].tolist()
            rec={"active_ids":active,"source_ids":[],"sham":None}
            if len(active)!=16:
                valid_base[mask]=False
                case_meta.append(rec)
                continue
            src=source_coherent_group(active,ranking,cosines,4)
            ids,weights,receipt=select_continuous_sham(
                active,src,z[mask],phys,cosines,reference["grad_down"][visit][mask])
            rec["source_ids"]=src
            rec["sham"]=receipt
            source[mask]=(z[mask,src]@dec[src])*nonhub
            sham[mask]=(weights@dec[ids])*nonhub
            if not receipt["selected_candidate_quality_valid"]:
                valid_base[mask]=False
            case_meta.append(rec)
        paired,dose=equalize_removal(source,sham,down)
        ratio=torch.tensor(dose["shared_output_L2"])/torch.tensor(
            dose["raw_source_output_L2"]).clamp_min(1e-10)
        min_relative=torch.minimum(min_relative,ratio)
        valid_base &= torch.tensor(
            [dose["positive_matched_dose"][m] and
             dose["dose_match_within_1pct"][m] for m in range(MASKS)],dtype=torch.bool
        )
        for tag,fraction in zip(TAGS,DOSES):
            variants[f"source_d{tag}"][visit]=full[visit]-fraction*paired["source"]
            variants[f"sham_d{tag}"][visit]=full[visit]-fraction*paired["sham"]
        geometry[str(visit)]={
            "native_selected_units":sorted(nodes),
            "original_graph_hubs":sorted(hubs.get(visit,[])),
            "case_meta":case_meta,
            "physical_pair_dose":dose,
            "full_affine_physical_output_L2":(full[visit]@down.T).norm(dim=1).tolist(),
        }
    # One eligibility policy for all five doses, with a positive scaled
    # physical output-dose floor, and all-visit validity retained.
    available={}
    relative={}
    for tag,fraction in zip(TAGS,DOSES):
        valid=valid_base & (min_relative*fraction>1e-6)
        available[tag]=valid.tolist()
        relative[tag]=torch.where(valid,min_relative*fraction,
                                  torch.zeros_like(min_relative)).tolist()
        for visit in sites:
            for side in ("source","sham"):
                mode=f"{side}_d{tag}"
                delta=variants[mode][visit]-full[visit]
                variants[mode][visit]=full[visit]+delta*valid[:,None]
    traces={"clean":clean,"hubs":lesion}
    for mode,vecs in variants.items():
        donors={v:{"native":t} for v,t in vecs.items()}
        with torch.no_grad():
            traces[mode]=traced_forward(
                model,x,mode="native",orientation=orientation,
                donors=donors,gates=gates
            )
    for tag in TAGS:
        invalid=~torch.tensor(available[tag])
        a=traces[f"source_d{tag}"]["logits"][invalid]
        b=traces[f"sham_d{tag}"]["logits"][invalid]
        baseline=traces["full_affine"]["logits"][invalid]
        if not (torch.allclose(a,b,atol=1e-6) and
                torch.allclose(a,baseline,atol=1e-6)):
            raise AssertionError("failed original source sham produced a nonzero contrast")
    return {
        "schema":SCHEMA,"source_group":group,
        "source_training_histories":histories,
        "focal_mask_order":list(range(MASKS)),
        "clean_chosen_orientation":orientation.tolist(),
        "labels_used_for_sham_selection":False,
        "site_geometry":geometry,
        "available_per_dose_all_visits":available,
        "min_site_relative_physical_dose_per_case":relative,
        "logits":{m:t["logits"].tolist() for m,t in traces.items()},
    }


def run(index,source,out,*,reset=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _,lock_sha=verify_lock()
    if index not in ROOTS:
        raise PermissionError("only original eight historical native source models")
    world=worlds()[index]
    out.mkdir(parents=True,exist_ok=True)
    arms={}
    for arm in ("composed","additive"):
        prepared=load_source_arm(index,arm,source,reset=reset)
        prepared["arm"]=arm
        if prepared["world"]["world_sha256"]!=world["world_sha256"]:
            raise AssertionError("native source training model remapped")
        correct={m:0 for m in MODES}
        valid={t:0 for t in TAGS}
        correct_valid_base={t:0 for t in TAGS}
        sham_c2w={t:0 for t in TAGS}
        sham_w2c={t:0 for t in TAGS}
        source_c2w={t:0 for t in TAGS}
        dose_values={t:[] for t in TAGS}
        raws=[]
        for group in CAL:
            inputs=prepared["catalog"]["train"][group*MASKS:(group+1)*MASKS]
            rec=group_measure(prepared,group,inputs)
            truth=labels(as_tensor(inputs),world,control=(arm=="additive")).tolist()
            prediction={}
            for m in MODES:
                pred=[int(x[1]>x[0]) for x in rec["logits"][m]]
                prediction[m]=pred
                correct[m]+=sum(int(p==y) for p,y in zip(pred,truth))
            for tag in TAGS:
                for m in range(MASKS):
                    if not rec["available_per_dose_all_visits"][tag][m]:
                        continue
                    valid[tag]+=1
                    baseline=prediction["full_affine"][m]==truth[m]
                    ctrl=prediction[f"sham_d{tag}"][m]==truth[m]
                    sourceok=prediction[f"source_d{tag}"][m]==truth[m]
                    correct_valid_base[tag]+=int(baseline)
                    sham_c2w[tag]+=int(baseline and not ctrl)
                    sham_w2c[tag]+=int(not baseline and ctrl)
                    source_c2w[tag]+=int(baseline and not sourceok)
                    dose_values[tag].append(
                        rec["min_site_relative_physical_dose_per_case"][tag][m]
                    )
            rec["arm"]=arm
            rec["teacher_source_truth_labels"]=truth
            rec["class_predictions"]=prediction
            raws.append(rec)
        content=b"".join(encoded(row) for row in raws)
        path=out/f"{arm}.nec03z_cal_source.jsonl"
        path.write_bytes(content)
        arms[arm]={
            "model_sha256":prepared["model_sha256"],
            "sae_sha256":prepared["sae_sha256"],
            "source_rank_sha256":prepared["rank_sha256"],
            "old_source_receipt_sha256":prepared["source_report_sha256"],
            "original_source_candidate_present":bool(prepared["source_candidate"]),
            "original_arm_source_eligible":prepared["original_arm_source_eligible"],
            "source_groups":list(CAL),
            "source_cases":CASES,
            "correct":correct,
            "dose_calibration":{
                t:{
                    "dose_fraction":NAME_TO_DOSE[t],
                    "available_original_source_cases":valid[t],
                    "baseline_correct_matched_cases":correct_valid_base[t],
                    "sham_correct_to_wrong":sham_c2w[t],
                    "sham_wrong_to_correct":sham_w2c[t],
                    "source_correct_to_wrong":source_c2w[t],
                    "sham_inertness_damage_fraction":(
                        sham_c2w[t]/correct_valid_base[t]
                        if correct_valid_base[t] else None
                    ),
                    "median_min_site_relative_dose":physical_median(dose_values[t]),
                } for t in TAGS
            },
            "raw_file":path.name,
            "raw_sha256":sha256(content).hexdigest(),
        }
        if model_digest(prepared["model"])!=prepared["model_sha256"]:
            raise AssertionError("calibration modified native model")
    report={
        "schema":SCHEMA,"world_index":index,"world_id":world["world_id"],
        "world_sha256":world["world_sha256"],
        "source_registry_sha256":registration()["registry_sha256"],
        "old_four_input_confirmation_sha256":old_registry()["confirmation_full_spec_sha256"],
        "source_manifest_sha256":context_catalog(world)[1]["source_sha256"],
        "source_lock_sha256":lock_sha,
        "calibration_source_groups_scored":list(CAL),
        "sealed_analysis_source_groups_scored":[],
        "source_cases_per_native_teacher":CASES,
        "original_paired_source_eligible":index in ELIGIBLE,
        "original_ITT_zero":index in ITT_ZERO,
        "arms":arms,
        "new_native_SAE_training_updates":0,
        "development_eval_scored":0,"reserved_scored":0,
        "recruited_worlds_scored":0,"embargoed_worlds_scored":0,
        "old_four_input_confirmation_scored":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"nec03z_cal_source_root_report.json").write_bytes(encoded(report))
    print("NEC03Z_FROZEN_CALIBRATION_SOURCE_ROOT_SCORED",world["world_id"])
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--world-index",required=True,type=int)
    parser.add_argument("--source-root",required=True,type=Path)
    parser.add_argument("--reset-root",type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    run(args.world_index,args.source_root,args.out,reset=args.reset_root)
