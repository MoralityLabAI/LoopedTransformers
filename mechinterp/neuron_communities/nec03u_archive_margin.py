"""NEC-03U: read-only immutable NEC03T source logit-margin audit.

Pure standard library. Reconstruct every teacher-signed logit margin from
the already-scored NEC03T source-training JSONL archives, verify every
old report, independent auditor, raw SHA256 and original ITT aggregate.
Never import/load Torch models, update weights or score new contexts.

All margin endpoints are RETROSPECTIVE, not registered confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha1, sha256
import json
import math
from pathlib import Path
import statistics

SCHEMA="NEC-03U/read-only-eight-root-archived-margin-audit/v1"
LOCK="NEC03U_ARCHIVE_LOCK.json"
ROOTS=tuple(range(8))
ELIGIBLE=(0,1,2,4,6,7)
ITT_ZERO=(3,5)
PANEL=tuple(range(24,32))
ARMS=("composed","additive")
MODES=(
    "clean","hubs","native_nonhub","full_affine",
    "contrib16_only","ablate_orig_source","ablate_orig_sham",
    "ablate_wide_source","ablate_wide_sham",
    "ablate_hash_source","ablate_hash_sham",
)
BINS=("negative","zero_to_2","2_to_5","5_to_10","10_or_more")
EPS=1e-7


def digest(b:bytes)->str:
    return sha256(b).hexdigest()


def file_sha(path:Path)->str:
    return digest(path.read_bytes())


def canonical_jsonl(items):
    return b"".join((json.dumps(item,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
                    for item in items)


def verify_lock():
    path=Path(__file__).with_name(LOCK)
    cfg=json.loads(path.read_text())
    exact={
        "schema":"NEC-03U/archived-source-margin-readonly-method/v1",
        "original_nec03t_run":44913605626,
        "original_world_root_indices":list(ROOTS),
        "original_source_group_indices":list(PANEL),
        "source_histories_per_arm":256,
        "paired_model_arms":list(ARMS),
        "original_eligible_root_indices":list(ELIGIBLE),
        "original_fixed_zero_roots":list(ITT_ZERO),
        "original_intervention_modes":list(MODES),
        "margin_bins":list(BINS),
        "margin_bin_boundaries_logit_units":[0.0,2.0,5.0,10.0],
        "signed_margin":"logits[true_label]-logits[1-true_label]",
        "retrospective_root_contrast":"mean(sham-source teacher-signed margin) composed minus additive; fixed 3,5 zero; eight-root mean",
        "correctness_tie_policy":"argmax first class wins: predict 1 only if logit1>logit0",
        "original_accuracy_primary_effect":1/2048,
        "posthoc_margin_pvalue_computed":False,
        "native_or_SAE_training_updates":0,
        "model_forwards":0,
        "development_eval_scored":0,
        "reserved_scored":0,
        "new_recruit_worlds_scored":0,
        "five_input_embargoed_scored":0,
        "four_input_confirmation_scored":0,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(cfg.get(k)!=v for k,v in exact.items()):
        raise AssertionError("read-only margin population or method changed")
    assert set(cfg["artifact_ids"])==set(map(str,ROOTS))|{"aggregate"}
    assert set(cfg["roots"])==set(map(str,ROOTS))
    assert len(cfg["original_aggregate_sha256"])==64
    for i in ROOTS:
        pin=cfg["roots"][str(i)]
        assert set(pin)=={
            "world_sha256","source_lock_sha256",
            "report_sha256","audit_sha256",
            "composed_jsonl_sha256","additive_jsonl_sha256"}
        assert all(len(v)==64 for v in pin.values())
    root=Path(__file__).resolve().parents[2]
    for name,pin in cfg["git_blob_sha1"].items():
        raw=(root/name).read_bytes()
        actual=sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()
        if actual!=pin:
            raise AssertionError("NEC03U retrospective auditor source identity changed "+name)
    return cfg,file_sha(path)


def finite(v, *, what="margin"):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
        raise AssertionError("nonfinite or nonnumeric "+what)
    return float(v)


def calc_margin(pair,label):
    if (not isinstance(pair,list) or len(pair)!=2
        or label not in (0,1)):
        raise AssertionError("incorrect original binary logits or teacher label")
    a,b=(finite(x,what="raw logit") for x in pair)
    return (b-a) if label==1 else (a-b)


def predict(pair):
    if not isinstance(pair,list) or len(pair)!=2:
        raise AssertionError("invalid binary logit pair")
    a,b=(finite(x,what="raw logit") for x in pair)
    return int(b>a)


def margin_bin(m):
    m=finite(m)
    if m<0: return "negative"
    if m<2: return "zero_to_2"
    if m<5: return "2_to_5"
    if m<10:return "5_to_10"
    return "10_or_more"


def stats(values):
    if not values:
        return {"count":0,"mean":None,"median":None,"mean_abs":None,
                "minimum":None,"maximum":None,"positive":0,"negative":0,"zero":0}
    vals=[finite(v) for v in values]
    return {
        "count":len(vals),
        "mean":statistics.fmean(vals),
        "median":statistics.median(vals),
        "mean_abs":statistics.fmean(map(abs,vals)),
        "minimum":min(vals),"maximum":max(vals),
        "positive":sum(v>EPS for v in vals),
        "negative":sum(v<(-EPS) for v in vals),
        "zero":sum(abs(v)<=EPS for v in vals),
    }


def verify_dose_receipt(site, mask, *, valid):
    if set(site["budget"])!={"orig","wide"}:
        raise AssertionError("missing source ablation family")
    for name in ("orig","wide"):
        group=site["budget"][name]
        code_ids=group["active_ids"][mask]
        if len(code_ids)!=len(set(code_ids)) or any(not isinstance(v,int) or v not in range(128)
                                                     for v in code_ids):
            raise AssertionError("raw TopK active IDs not a valid set")
        positive=group["exact_positive16_codes"][mask]
        if positive!=(len(code_ids)==16):
            raise AssertionError("active support flag changed")
        source=group["source_ids"][mask]
        sham=group["angular_sham_ids"][mask]
        k=group["k"]
        if not (1<=k<=8):
            raise AssertionError("invalid positive active ablation budget")
        n=group["candidate_count"][mask]
        gap=group["angle_gap"][mask]
        if positive:
            if not (len(source)==len(sham)==k
                    and len(set(source))==len(set(sham))==k
                    and not (set(source)&set(sham))
                    and set(source+sham)<=set(code_ids)
                    and n==math.comb(16-k,k)):
                raise AssertionError("original disjoint source/sham subset or candidate count changed")
            finite(gap,what="sham cosine gap")
            if gap<0:
                raise AssertionError("negative angular gap")
        elif source or sham or n!=0 or gap is not None:
            raise AssertionError("unsupported active TopK input had a sham")
        dose=group["angular_removal_dose"]
        physical={key:dose["raw_dose"][key][mask] for key in ("source","sham")}
        minimum=min(finite(x,what="raw dose") for x in physical.values())
        shared=finite(dose["shared_dose"][mask],what="shared dose")
        if abs(minimum-shared)>max(1e-5,1e-5*abs(minimum)):
            raise AssertionError("matched physical source/sham dose changed")
        if dose["nonzero_shared_dose"][mask]!=(shared>1e-6):
            raise AssertionError("nonzero ablation dose status changed")
        for key in physical:
            gain=finite(dose["gains"][key][mask],what="attenuation gain")
            if gain<0 or gain>1.000001 or physical[key]<0:
                raise AssertionError("negative physical dose or amplified donor")
            if shared>1e-6 and abs(gain*physical[key]-shared)>max(1e-4,shared*.01001):
                raise AssertionError("matched original physical dose differs")
        if valid[name] and (not positive or gap>.250001 or shared<=1e-6
                            or not dose["physical_dose_equal_within_1pct"][mask]):
            raise AssertionError("invalid source control marked available")
    # Hash null only at original budget, with its own equal physical dose
    hashdose=site["budget"]["orig"]["hash_removal_dose"]
    hraw=hashdose["raw_dose"]
    hshared=finite(hashdose["shared_dose"][mask],what="hash shared dose")
    if abs(hshared-min(finite(hraw[k][mask]) for k in ("source","sham")))>1e-5:
        raise AssertionError("archive hash control physical L2 mismatch")


def read_root(index,folder,pinned,original_root,raw_records):
    rpath=folder/"nec03t_source_necessity_report.json"
    apath=folder/"independent_nec03t_source_necessity_audit.json"
    if file_sha(rpath)!=pinned["report_sha256"] or file_sha(apath)!=pinned["audit_sha256"]:
        raise AssertionError("NEC03T original report or independent audit content altered")
    report=json.loads(rpath.read_text())
    audit=json.loads(apath.read_text())
    if (report["world_id"]!=f"dev5-{index:03d}"
        or report["world_index"]!=index
        or audit["world_index"]!=index
        or report["world_sha256"]!=pinned["world_sha256"]
        or audit["world_sha256"]!=pinned["world_sha256"]
        or original_root["world_sha256"]!=pinned["world_sha256"]
        or report["source_lock_sha256"]!=pinned["source_lock_sha256"]
        or audit["source_lock_sha256"]!=pinned["source_lock_sha256"]
        or audit["original_report_sha256"]!=file_sha(rpath)
        or original_root["source_report_sha256"]!=file_sha(rpath)
        or original_root["independent_audit_sha256"]!=file_sha(apath)
        or not original_root["independent_audit_complete"]
        or report["source_groups_scored"]!=list(PANEL)
        or report["source_training_cases_per_arm"]!=256
        or report["original_paired_source_eligible"]!=(index in ELIGIBLE)
        or report["original_ITT_source_failure"]!=(index in ITT_ZERO)
        or audit["source_cases_per_arm"]!=256
        or set(report["arms"])!=set(ARMS) or set(audit["arms"])!=set(ARMS)
        or any(report[x]!=0 or audit[x]!=0 for x in (
            "new_development_eval_groups_scored","reserved_groups_scored",
            "newly_recruited_worlds_scored","embargoed_worlds_scored",
            "old_four_input_confirmation_worlds_scored"))
        or report["new_native_or_SAE_training_updates"]!=0
        or report["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"
        or audit["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"):
        raise AssertionError("source ancestry, outcome-free conditions or original eight-root denominator changed")
    if not all(audit[key] for key in (
        "all_original_source_groups_and_true_labels_verified",
        "all_native_and_SAE_checkpoint_provenance_verified",
        "full_affine_source_background_independently_reconstructed",
        "all_disjoint_ranked_shams_and_hash_random_controls_recomputed",
        "all_matched_physical_ablation_doses_and_invalid_cases_replayed",
        "all_eleven_full_recurrent_outcomes_independently_recomputed")):
        raise AssertionError("original independent full-source audit incomplete")
    arms={}
    row_groups={}
    root_case_delta={}
    for arm in ARMS:
        old=report["arms"][arm]
        verified=audit["arms"][arm]
        root_summary=original_root["arms"][arm]
        if old["arm"]!=arm or old["source_histories"]!=256 or old["source_groups_scored"]!=list(PANEL):
            raise AssertionError("old source arm was missing a complete training-source cohort")
        raw=folder/old["raw_file"]
        if (old["raw_file"]!=f"{arm}.nec03t_source_necessity.jsonl"
            or file_sha(raw)!=pinned[f"{arm}_jsonl_sha256"]
            or file_sha(raw)!=old["raw_sha256"]
            or verified["raw_sha256"]!=old["raw_sha256"]):
            raise AssertionError("prior raw JSONL was altered")
        if (old["correct"]!=verified["correct"]
            or old["correct"]!=root_summary["correct"]
            or old["source_ranked_necessity_correct_count"]!=verified["necessity_correct_count"]
            or old["source_model_sha256"]!=verified["source_model_sha256"]
            or old["source_sae_checkpoint_sha256"]!=verified["source_sae_checkpoint_sha256"]
            or old["valid_angular_case_count"]!=verified["matched_angular_cases"]
            or old["valid_hash_case_count"]!=verified["matched_hash_cases"]):
            raise AssertionError("raw report differs from original independent source audit")
        records=[json.loads(line) for line in raw.read_text().splitlines()]
        if len(records)!=len(PANEL):
            raise AssertionError("missing old original source group")
        by_mode=Counter()
        margins={name:[] for name in MODES}
        deltas=[];full_base=[];source_loss=[];sham_loss=[]
        groups=[];strata={name:{"deltas":[],"valid":0} for name in BINS}
        original_avail=0
        cases_by_id={}
        seen=set()
        for g,rec in zip(PANEL,records):
            if (rec["schema"]!="NEC-03T/source-only-full-affine-necessity-ablation/v1"
                or rec["source_group"]!=g or rec["arm"]!=arm
                or rec["labels_used_for_routing"] is not False
                or rec["focal_mask_order"]!=list(range(32))
                or set(rec["logits"])!=set(MODES)
                or set(rec["class_predictions"])!=set(MODES)
                or set(rec["valid_angular_all_visits"])!={"orig","wide"}
                or not isinstance(rec["site_geometry"],dict)):
                raise AssertionError("read-only source transcript scope/modes violated")
            y=rec["true_teacher_labels"]
            histories=rec["input_histories"]
            if len(y)!=32 or len(histories)!=32:
                raise AssertionError("incomplete source five-bit focal group")
            if len({tuple(row) for row in histories})!=32:
                raise AssertionError("source group lacks 32 unique complete focal histories")
            if any(len(row)!=12 or any(bit not in (0,1) for bit in row) for row in histories):
                raise AssertionError("source history outside the original twelve-bit source task")
            if any(len(rec["logits"][name])!=32 or
                   len(rec["class_predictions"][name])!=32 for name in MODES):
                raise AssertionError("incomplete original binary-logit source mode")
            if any(len(rec["valid_angular_all_visits"][b])!=32 for b in ("orig","wide")):
                raise AssertionError("invalid-case full-denominator source mask omitted")
            for site in rec["site_geometry"].values():
                if len(site["full_affine_physical_dose"])!=32:
                    raise AssertionError("missing original physical full-affine dose")
                for m in range(32):
                    verify_dose_receipt(site,m,valid={
                        b:rec["valid_angular_all_visits"][b][m]
                        for b in ("orig","wide")})
            for m in range(32):
                case_id=(g,m)
                if case_id in seen:
                    raise AssertionError("source group/focal mask double counted")
                seen.add(case_id)
                truth=y[m]
                if truth not in (0,1):
                    raise AssertionError("original source teacher truth invalid")
                cur={}
                for mode in MODES:
                    pair=rec["logits"][mode][m]
                    prediction=predict(pair)
                    margin=calc_margin(pair,truth)
                    if prediction!=rec["class_predictions"][mode][m]:
                        raise AssertionError("raw original class prediction no longer agrees with logits")
                    by_mode[mode]+=int(prediction==truth)
                    cur[mode]=margin
                    margins[mode].append(margin)
                is_valid=rec["valid_angular_all_visits"]["orig"][m]
                if not isinstance(is_valid,bool):
                    raise AssertionError("original angular availability not boolean")
                if not is_valid and abs(cur["ablate_orig_sham"]-cur["ablate_orig_source"])>1e-6:
                    raise AssertionError("invalid-case source/sham produced nonzero margin effect")
                if not rec["valid_angular_all_visits"]["wide"][m]:
                    if abs(cur["ablate_wide_sham"]-cur["ablate_wide_source"])>1e-6:
                        raise AssertionError("invalid widened source/sham produced nonzero margin effect")
                if not rec["valid_hash_all_visits"][m]:
                    if abs(cur["ablate_hash_sham"]-cur["ablate_hash_source"])>1e-6:
                        raise AssertionError("invalid hash source/sham produced nonzero margin effect")
                original_avail+=int(is_valid)
                difference=cur["ablate_orig_sham"]-cur["ablate_orig_source"]
                full_base.append(cur["full_affine"])
                deltas.append(difference)
                source_loss.append(cur["full_affine"]-cur["ablate_orig_source"])
                sham_loss.append(cur["full_affine"]-cur["ablate_orig_sham"])
                bin_name=margin_bin(cur["full_affine"])
                strata[bin_name]["deltas"].append(difference)
                strata[bin_name]["valid"]+=int(is_valid)
                cases_by_id[case_id]={
                    "world_id":report["world_id"],"arm":arm,"source_group":g,
                    "focal_mask":m,"truth":truth,
                    "frozen_original_history":histories[m],
                    "full_affine_margin_bin":bin_name,
                    "valid_angular_orig":is_valid,
                    "valid_angular_wide":rec["valid_angular_all_visits"]["wide"][m],
                    "valid_hash":rec["valid_hash_all_visits"][m],
                    "signed_margins":cur,
                    "retrospective_sham_minus_source_margin":difference,
                }
        if len(seen)!=256 or original_avail!=old["valid_angular_case_count"]["orig"]:
            raise AssertionError("original 256 source histories or source/sham availability changed")
        if any(by_mode[mode]!=old["correct"][mode] for mode in MODES):
            raise AssertionError("recalculated correctness does not match old NEC03T scorer")
        if (by_mode["ablate_orig_sham"]-by_mode["ablate_orig_source"]
            != old["source_ranked_necessity_correct_count"]["orig"]):
            raise AssertionError("original NEC03T accuracy necessity contrast changed")
        bin_reports={
            k:{"count":len(v["deltas"]),
               "valid_matched_controls":v["valid"],
               "sham_minus_source_margin":stats(v["deltas"])}
            for k,v in strata.items()
        }
        if sum(row["count"] for row in bin_reports.values())!=256:
            raise AssertionError("baseline margin strata lost source cases")
        arms[arm]={
            "original_correct_counts":dict(by_mode),
            "n_cases":256,
            "original_angular_valid_cases":original_avail,
            "original_accuracy_necessity_case_difference":
                by_mode["ablate_orig_sham"]-by_mode["ablate_orig_source"],
            "retrospective_sham_minus_source_margin":stats(deltas),
            "full_affine_teacher_signed_margin":stats(full_base),
            "source_removal_loss_from_full_affine":stats(source_loss),
            "sham_removal_loss_from_full_affine":stats(sham_loss),
            "strata_frozen_full_affine_margin":bin_reports,
            "old_raw_jsonl_sha256":file_sha(raw),
        }
        row_groups[arm]=[cases_by_id[i] for i in sorted(cases_by_id)]
        root_case_delta[arm]={i:cases_by_id[i]["retrospective_sham_minus_source_margin"]
                              for i in sorted(cases_by_id)}
    if [x["frozen_original_history"] for x in row_groups["composed"]] != [
        x["frozen_original_history"] for x in row_groups["additive"]]:
        raise AssertionError("original paired teacher source input histories are no longer paired")
    per_case_pair=[
        root_case_delta["composed"][key]-root_case_delta["additive"][key]
        for key in sorted(root_case_delta["composed"])
    ]
    raw_accuracy=(
        arms["composed"]["original_accuracy_necessity_case_difference"]-
        arms["additive"]["original_accuracy_necessity_case_difference"])/256
    if (abs(raw_accuracy-original_root["unselected_raw_primary_DID"])>1e-12
        or abs(raw_accuracy-report["raw_primary_source_necessity_DID"])>1e-12
        or abs((raw_accuracy if index in ELIGIBLE else 0)-
            original_root["root_primary_necessity_DID"])>1e-12):
        raise AssertionError("original NEC03T root-level accuracy ITT effect changed")
    case_rows=row_groups["composed"]+row_groups["additive"]
    result={
        "world_id":report["world_id"],"world_index":index,
        "world_sha256":pinned["world_sha256"],
        "original_paired_source_eligible":index in ELIGIBLE,
        "original_primary_ITT_zero":index in ITT_ZERO,
        "original_nec03t_primary_accuracy_effect":raw_accuracy if index in ELIGIBLE else 0.,
        "original_source_report_sha256":file_sha(rpath),
        "original_independent_source_audit_sha256":file_sha(apath),
        "arms":arms,
        "paired_teacher_casewise_margin_difference":stats(per_case_pair),
        "retrospective_root_mean_margin_DID":
            stats(per_case_pair)["mean"] if index in ELIGIBLE else 0.,
        "retrospective_root_median_margin_DID":
            stats(per_case_pair)["median"] if index in ELIGIBLE else 0.,
    }
    result["read_only_case_receipt_sha256"]=digest(canonical_jsonl(case_rows))
    return result,case_rows


def audit(root:Path,old_aggregate:Path,out:Path):
    cfg,lock_sha=verify_lock()
    if file_sha(old_aggregate)!=cfg["original_aggregate_sha256"]:
        raise AssertionError("immutable old NEC03T eight-root aggregate altered")
    parent=json.loads(old_aggregate.read_text())
    if (parent["schema"]!="NEC-03T/eight-root-source-affine-necessity-ITT/v1"
        or not parent["all_independent_root_audits_pass"]
        or parent["independently_audited_root_count"]!=8
        or parent["registered_root_count"]!=8
        or parent["original_source_qualified_roots"]!=list(ELIGIBLE)
        or parent["original_source_ineligible_ITT_zero_roots"]!=list(ITT_ZERO)
        or parent["source_groups_scored"]!=list(PANEL)
        or parent["source_histories_per_model_arm"]!=256
        or parent["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"
        or any(parent[k]!=0 for k in (
            "new_development_eval_groups_scored","reserved_groups_scored",
            "newly_recruited_worlds_scored","embargoed_worlds_scored",
            "old_four_input_confirmation_worlds_scored"))
        or set(parent["source_correct_counts"])!=set(ARMS)
        or len(parent["roots"])!=8
        or abs(parent["primary_macro_mean_necessity_effect"]-(1/2048))>1e-12):
        raise AssertionError("old independently audited NEC03T population or primary changed")
    original_paths={p.name for p in root.iterdir() if p.is_dir()}
    expected={f"nec03t-source-root-{i}" for i in ROOTS}
    if original_paths!=expected:
        raise AssertionError("missing or unexpected source root artifacts")
    summaries=[]
    cases=[]
    for i in ROOTS:
        prior=parent["roots"][i]
        if prior["world_index"]!=i:
            raise AssertionError("original ITT root sequence changed")
        entry,rows=read_root(
            i,root/f"nec03t-source-root-{i}",
            cfg["roots"][str(i)],prior,cases)
        summaries.append(entry)
        cases.extend(rows)
    if len(cases)!=4096 or len(summaries)!=8:
        raise AssertionError("archived 8x2x256 source histories incomplete")
    reconstructed={
        arm:{mode:sum(s["arms"][arm]["original_correct_counts"][mode] for s in summaries)
             for mode in MODES}
        for arm in ARMS}
    if reconstructed!=parent["source_correct_counts"]:
        raise AssertionError("parent eight-root correct-count aggregate differs from logits")
    original_effects=[s["original_nec03t_primary_accuracy_effect"] for s in summaries]
    if (any(abs(a-b)>1e-12 for a,b in zip(original_effects,parent["primary_root_necessity_effects"]))
        or abs(sum(original_effects)/8-1/2048)>1e-12):
        raise AssertionError("parent original 8-root accuracy sham contrast changed")
    margin_root_effects=[r["retrospective_root_mean_margin_DID"] for r in summaries]
    medians=[r["retrospective_root_median_margin_DID"] for r in summaries]
    result={
        "schema":SCHEMA,
        "source_lock_sha256":lock_sha,
        "original_nec03t_run":44913605626,
        "original_aggregate_sha256":file_sha(old_aggregate),
        "original_aggregate_independent_root_audits_passed":True,
        "all_original_eight_root_raw_archive_sha256_verified":True,
        "all_4096_archived_source_label_correctness_predictions_recomputed":True,
        "no_torch_model_or_SAE_loaded":True,
        "new_model_forwards":0,
        "original_world_roots":list(ROOTS),
        "original_source_eligible_roots":list(ELIGIBLE),
        "original_source_ITT_zero_roots":list(ITT_ZERO),
        "all_sixteen_paired_model_arms":True,
        "source_groups_scored":list(PANEL),
        "source_histories_per_arm":256,
        "total_original_source_case_arm_rows":len(cases),
        "original_primary_accuracy_ITT_effect":sum(original_effects)/8,
        "original_primary_correctness_p":"original archived NEC03T exploratory root sign-flip p=1, not re-tested",
        "retrospective_margin_method":"teacher signed binary logit delta, equal source root weighted",
        "posthoc_margin_pvalue_computed":False,
        "retrospective_root_mean_margin_effects":margin_root_effects,
        "retrospective_root_median_margin_effects":medians,
        "retrospective_eight_root_mean_margin_DID_logit_units":statistics.fmean(margin_root_effects),
        "retrospective_root_effect_median_logit_units":statistics.median(margin_root_effects),
        "retrospective_root_effect_range_logit_units":[min(margin_root_effects),max(margin_root_effects)],
        "retrospective_root_signs":{
            "positive":sum(v>EPS for v in margin_root_effects),
            "negative":sum(v<-EPS for v in margin_root_effects),
            "zero":sum(abs(v)<=EPS for v in margin_root_effects),
        },
        "logit_scale_caution":"raw native logit units may differ by model/teacher; root-level equal weighting is descriptive, not scale calibrated",
        "all_bins_predeclared_before_archived_margin_audit":list(BINS),
        "scientific_evidence":"retrospective already-trained source-only, descriptive",
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "native_or_SAE_training_updates":0,
        "new_development_eval_groups_scored":0,
        "new_reserved_groups_scored":0,
        "new_recruited_worlds_scored":0,
        "new_five_input_embargoed_worlds_scored":0,
        "new_four_input_confirmation_worlds_scored":0,
        "root_summaries":summaries,
    }
    out.mkdir(parents=True,exist_ok=True)
    case_data=canonical_jsonl(cases)
    (out/"nec03u_archived_source_margin_cases.jsonl").write_bytes(case_data)
    result["archived_margin_case_records_sha256"]=digest(case_data)
    (out/"nec03u_readonly_margin_aggregate.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("NEC03U_ARCHIVED_READ_ONLY_MARGIN_AUDIT_PASS",
          result["retrospective_eight_root_mean_margin_DID_logit_units"],
          "original_accuracy",result["original_primary_accuracy_ITT_effect"],
          "root_audits",len(summaries),"case_rows",len(cases))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--original-aggregate",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    audit(a.root,a.original_aggregate,a.out)


if __name__=="__main__":
    main()
