"""NEC-03I: pure-stdlib forensic replay of NEC-03H signed causal outcomes.

Only the eight original dev-008..015 V2 artifacts are admissible.
Per-world effects remain the independent population; all 48 contexts and
16 focal counterfactuals are correlated measurements within each world.
Every checksum, truth label, source-selected TopK route, direct physical
attenuation and registered signed effect is checked before aggregation.

This is a read-only, posthoc diagnostic. It trains no models and reads
neither confirmation worlds nor reserved context outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean

from mechinterp.neuron_communities.nec03g_worlds import (
    additive_control_truth, context_catalog, registry, worlds,
)

SCHEMA = "NEC-03I/read-only-effect-direction-and-dose-audit/v1"
PARENT = "NEC-03H/source-site-fallback-900steps-development-v2"
PAIRS = (("native", "sham"), ("native", "topk"), ("topk", "topk_alt"))
INDICES = tuple(range(8, 16))


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AssertionError("expected manifest object: " + str(path))
    return value


def file_sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_bytes(path: Path, expected: str) -> None:
    if len(expected) != 64 or file_sha(path) != expected:
        raise AssertionError("source byte hash mismatch: " + str(path))


def clean_hits(pairs, labels) -> list[int]:
    if len(pairs) != 16 or len(labels) != 16:
        raise AssertionError("incomplete correlated 16-state quartet")
    hits = []
    for values, truth in zip(pairs, labels):
        if len(values) != 2 or not all(math.isfinite(float(x)) for x in values):
            raise AssertionError("nonfinite or missing binary logits")
        if truth not in (0, 1):
            raise AssertionError("invalid truth label")
        prediction = int(float(values[1]) > float(values[0]))
        hits.append(int(prediction == truth))
    return hits


def verify_feature_policy(policy: dict, ranks: list, count: int = 16) -> dict:
    """Verify each input chooses the frozen highest/lowest ranked ACTIVE codes."""
    if not policy:
        raise AssertionError("missing input-conditioned route")
    total = 0
    diversity = {}
    active_coverage = 0
    for key, samples in policy.items():
        visit = int(key)
        if not 0 <= visit < 12:
            raise AssertionError("route used outside the 12 recurrent visits")
        rank = ranks[visit]
        if len(rank) != 128 or set(rank) != set(range(128)):
            raise AssertionError("SAE source ranks are not a full permutation")
        position = {feature: j for j, feature in enumerate(rank)}
        top = samples["top_source_ranked_active"]
        bottom = samples["bottom_source_ranked_active"]
        actives = samples["all_active_features"]
        n = samples["selected_features_per_arm_per_visit"]
        if (len(top) != count or len(bottom) != count
                or len(actives) != count or not 1 <= n <= 8):
            raise AssertionError("feature policy violates per-site budget")
        chosen = set()
        for i in range(count):
            active = actives[i]
            if (len(active) != 16 or len(set(active)) != 16
                or any(x not in position for x in active)):
                raise AssertionError("input is not represented by 16 distinct active codes")
            ordered = sorted(active, key=lambda x: position[x])
            if top[i] != ordered[:n] or bottom[i] != list(reversed(ordered[-n:])):
                raise AssertionError("top/bottom SAE source rank selected incorrectly")
            if set(top[i]) & set(bottom[i]):
                raise AssertionError("TopK candidate/control feature overlap")
            chosen.update(top[i])
            total += n
            active_coverage += len(active)
        diversity[key] = len(chosen)
    return {"routed_features": total, "active_features_reported": active_coverage,
            "unique_selected_features_per_visit": diversity}


def verify_physical_dose(site: dict) -> list[tuple[float, float, bool]]:
    """Recompute gains and dose equality using the original displayed norms."""
    cols = ("full_left_dose", "full_right_dose", "shared_achievable_dose",
            "realized_left_dose", "realized_right_dose", "left_gain",
            "right_gain", "retained_fraction_of_full_left_dose",
            "relative_dose_mismatch", "matched_within_1pct", "shared_nonzero")
    if any(k not in site or len(site[k]) != 16 for k in cols):
        raise AssertionError("incomplete physical dose ledger")
    out = []
    for i in range(16):
        a = float(site["full_left_dose"][i])
        b = float(site["full_right_dose"][i])
        shared = float(site["shared_achievable_dose"][i])
        la = float(site["realized_left_dose"][i])
        rb = float(site["realized_right_dose"][i])
        ga = float(site["left_gain"][i])
        gb = float(site["right_gain"][i])
        ratio = float(site["retained_fraction_of_full_left_dose"][i])
        relative = abs(la-rb)/max(shared, 1e-6)
        if any(not math.isfinite(v) or v < 0 for v in (a,b,shared,la,rb,ga,gb,ratio,relative)):
            raise AssertionError("nonfinite/negative physical dose")
        if ga > 1 + 1e-6 or gb > 1 + 1e-6:
            raise AssertionError("donor amplification detected in attenuation-only protocol")
        if (abs(min(a,b)-shared) > 1e-4
            or abs(a*ga-shared) > 1e-4
            or abs(b*gb-shared) > 1e-4
            or abs(la-shared) > 1e-4
            or abs(rb-shared) > 1e-4):
            raise AssertionError("actual/registered physical output doses differ")
        if abs(relative-float(site["relative_dose_mismatch"][i])) > 1e-5:
            raise AssertionError("relative dose mismatch tampered")
        if bool(site["matched_within_1pct"][i]) != (relative<=.0100001):
            raise AssertionError("dose matching verdict differs")
        is_nonzero = shared > 1e-6
        if bool(site["shared_nonzero"][i]) != is_nonzero:
            raise AssertionError("positive dose indicator differs")
        expected_ratio=shared/a if a>1e-6 else 0.
        if abs(ratio-expected_ratio)>1e-4:
            raise AssertionError("fraction of original donor dose differs")
        out.append((shared,ratio,is_nonzero))
    return out


def audit_arm(folder: Path, world: dict, source: dict, arm: str) -> dict:
    report=read(folder / f"{arm}.report.json")
    if report["world_sha256"] != world["world_sha256"]:
        raise AssertionError("teacher uses different world")
    if report["source_manifest_sha256"]!=source["sha256"]:
        raise AssertionError("teacher uses different context split")
    native=report["train"]
    require_bytes(folder/native["checkpoint_name"],native["checkpoint_sha256"])
    if (not isinstance(native["final_model_sha256"],str)
        or len(native["final_model_sha256"])!=64):
        raise AssertionError("invalid frozen native weight identity")
    graph=folder/f"{arm}.community_source_freeze.json"
    require_bytes(graph, report["source_graph_freeze_sha256"])
    source_graph=read(graph)["source_discovery"]
    if source_graph["development_or_reserved_contexts_exposed_for_selection"]:
        raise AssertionError("source graph selected after reading evaluation")
    if report["candidate_status"]!="SOURCE_SELECTED":
        if report["raw_paired_records"] or report["pairs"]:
            raise AssertionError("absent source candidate has bogus scored pairs")
        return {"arm":arm,"status":"no_source_candidate","site_kind":"none",
                "source_graph_qualified":bool(source_graph["graph_has_qualifying_community"]),
                "clean_correct":report["clean"]["correct"],"pairs":{},"raw_row_count":0}
    freeze=folder/f"{arm}.source_freeze.json"
    require_bytes(freeze,report["source_freeze_sha256"])
    selection=read(freeze)
    if (selection["source_manifest_sha256"]!=source["sha256"]
        or selection["world_sha256"]!=world["world_sha256"]
        or selection["model_sha256"]!=native["final_model_sha256"]):
        raise AssertionError("source-selected model/SAE site mismatch")
    if selection["source_graph_candidate"]!=source_graph["candidate"]:
        raise AssertionError("native candidate changed after freeze")
    require_bytes(folder/f"{arm}.sae.pt",
                  selection["source_sae"]["source_sae_sha256"])
    if not selection["no_eval_or_reserved_contexts_used_for_selection"]:
        raise AssertionError("source ranks inspected heldout")
    raw=folder/report["raw_paired_context_file"]
    require_bytes(raw,report["raw_paired_context_sha256"])
    rows=[json.loads(x) for x in raw.read_text().splitlines()]
    if len(rows)!=report["raw_paired_records"] or len(rows)!=3*48:
        raise AssertionError("incomplete source-context × comparison roster")
    truths=(world["truth_table_16"] if arm=="composed"
            else additive_control_truth(world["truth_table_16"]))
    labels=[(truths>>i)&1 for i in range(16)]
    observed=set()
    pair_counts=defaultdict(lambda:Counter())
    context_effects=defaultdict(dict)
    focal_mask_effects=defaultdict(lambda:[0]*16)
    label_effects=defaultdict(lambda:Counter())
    site_samples=defaultdict(list)
    changed_case_doses=defaultdict(lambda:{"top_only":[],"alternative_only":[]})
    site_route_counts=defaultdict(lambda:Counter())
    for record in rows:
        name=tuple(record["pair"])
        ctx=int(record["source_context_index"])
        if (name not in PAIRS or ctx not in range(48)
                or (name,ctx) in observed):
            raise AssertionError("unregistered pair, context, or duplicate")
        observed.add((name,ctx))
        if record["world_id"]!=world["world_id"] or record["arm"]!=arm:
            raise AssertionError("paired logits belong to wrong teacher world")
        if ("".join(map(str,record["source_context_bits"])) !=
            source["contexts_ordered"][160+ctx]):
            raise AssertionError("paired logits accessed unregistered source context")
        if record["labels"] != labels or record["focal_mask_order"]!=list(range(16)):
            raise AssertionError("truth-function labels or focal mask order changed")
        clean=clean_hits(record["clean_logits"],labels)
        hubs=clean_hits(record["hubs_logits"],labels)
        left=clean_hits(record["left_logits"],labels)
        right=clean_hits(record["right_logits"],labels)
        for label,hits in (("clean",clean),("hubs",hubs),("left",left),("right",right)):
            pair_counts[name][label+"_errors"]+=16-sum(hits)
        delta=[a-b for a,b in zip(left,right)]
        context_effects[name][ctx]=sum(delta)
        for focal,(d,truth) in enumerate(zip(delta,labels)):
            focal_mask_effects[name][focal]+=d
            label_effects[name][truth]+=d
            if d>0: pair_counts[name]["left_only"]+=1
            if d<0: pair_counts[name]["right_only"]+=1
        if set(record["local_physical_doses"])!=set(record["active_features"]):
            raise AssertionError("ranked donor and local physical visits differ")
        verify_feature_policy(record["active_features"],selection["source_ranking_per_visit"])
        site_values=[]
        for visit,receipt in record["local_physical_doses"].items():
            reconstructed=verify_physical_dose(receipt)
            site_values.append(reconstructed)
            site_samples[name].extend(reconstructed)
            pol=record["active_features"][visit]
            for chosen in pol["top_source_ranked_active"]:
                site_route_counts[name].update((int(visit),int(f)) for f in chosen)
        for j,d in enumerate(delta):
            if d>0:
                changed_case_doses[name]["top_only"].extend(site[j][0] for site in site_values)
            elif d<0:
                changed_case_doses[name]["alternative_only"].extend(site[j][0] for site in site_values)
    if observed!={(pair,i) for pair in PAIRS for i in range(48)}:
        raise AssertionError("missing masked context")
    pairs={}
    for pair in PAIRS:
        official=next((p for p in report["pairs"] if tuple(p["pair"])==pair),None)
        if official is None:
            raise AssertionError("incomplete pair report")
        c=pair_counts[pair]
        expected={label:c[label+"_errors"] for label in ("clean","hubs","left","right")}
        if official["errors_of_768"]!=expected:
            raise AssertionError("summary class errors differ from raw logits")
        signed=c["right_errors"]-c["left_errors"]
        if signed!=official["left_minus_right_correct"]:
            raise AssertionError("reported signed effect differs from raw evidence")
        sample=site_samples[pair]
        ret=sorted(v[1] for v in sample)
        med=(ret[(len(ret)-1)//2]+ret[len(ret)//2])/2
        positive=sum(v[2] for v in sample)/len(sample)
        dose=official["physical_dose"]
        if (len(sample)!=dose["site_sample_count"]
            or abs(positive-dose["fraction_with_positive_shared_dose"])>1e-7
            or abs(med-dose["median_fraction_full_left_dose_preserved"])>1e-6):
            raise AssertionError("dose signal summary changed")
        eligible=positive>=.8 and med>=.25 and dose["fraction_matched_within_1pct"]>=.99
        if eligible!=dose["candidate_physically_comparable_and_signal_sufficient"]:
            raise AssertionError("dose eligibility altered")
        context=context_effects[pair]
        signed_ctx={"positive":sum(v>0 for v in context.values()),
                    "negative":sum(v<0 for v in context.values()),
                    "zero":sum(v==0 for v in context.values())}
        pairs["/".join(pair)]={
            "left_minus_right_correct":signed,
            "left_only_correct_cases":c["left_only"],
            "right_only_correct_cases":c["right_only"],
            "errors":expected,
            "eligible_physical_signal":eligible,
            "contexts_effect_positive_negative_zero":signed_ctx,
            "focal_mask_correctness_delta":focal_mask_effects[pair],
            "true_class_correctness_delta":dict(sorted(label_effects[pair].items())),
            "mean_common_direct_dose":mean(v[0] for v in sample),
            "median_fraction_left_dose_retained":med,
            "positive_shared_dose_fraction":positive,
            "mean_common_dose_on_left_only_correct":
                mean(changed_case_doses[pair]["top_only"]) if changed_case_doses[pair]["top_only"] else None,
            "mean_common_dose_on_right_only_correct":
                mean(changed_case_doses[pair]["alternative_only"]) if changed_case_doses[pair]["alternative_only"] else None,
            "unique_source_ranked_features_used":
                len(site_route_counts[pair]),
            "source_feature_routes_most_frequent":
                [{"visit":v,"feature":f,"selections":n}
                 for (v,f),n in site_route_counts[pair].most_common(10)],
        }
    if len(report["pairs"])!=len(PAIRS):
        raise AssertionError("unexpected multiple-comparison rows")
    return {"arm":arm,"status":"source_scored",
            "site_kind":report["source_site_kind"],
            "source_graph_qualified":bool(source_graph["graph_has_qualifying_community"]),
            "native_nodes":selection["source_graph_candidate"]["nodes"],
            "clean_correct":report["clean"]["correct"],
            "source_freeze_sha256":report["source_freeze_sha256"],
            "native_checkpoint_sha256":native["checkpoint_sha256"],
            "sae_checkpoint_sha256":selection["source_sae"]["source_sae_sha256"],
            "raw_pairs_sha256":report["raw_paired_context_sha256"],
            "pairs":pairs,"raw_row_count":len(rows)}


def audit_root(folder: Path, expected_index: int, lock_sha: str) -> dict:
    index=expected_index
    if index not in INDICES:
        raise AssertionError("opened/confirmation world not part of NEC-03I")
    world=read(folder/"world_spec.json")
    trusted=worlds()[index]
    if world!=trusted:
        raise AssertionError("world identity differs from old registered truth function")
    source=read(folder/"context_source_manifest.json")
    _,expected_source=context_catalog(trusted)
    if source!=expected_source:
        raise AssertionError("source context closure or reserved assignment changed")
    root=read(folder/"root_summary.json")
    independent=read(folder/"independent_source_audit.json")
    if (root["world_index"]!=index or root["cohort"]!="development"
        or root["schema"]!=PARENT
        or root["world_sha256"]!=world["world_sha256"]
        or root["world_source_sha256"]!=source["sha256"]
        or root["original_NEC03G_world_registry_sha256"]!=registry()["registry_sha256"]
        or root["v2_protocol_lock_sha256"]!=lock_sha
        or root["original_confirmation_worlds_evaluated"]!=0
        or root["reserved_context_groups_evaluated"]!=0
        or not root["full_registered_v2_recipe"]):
        raise AssertionError("wrong frozen NEC-03H population, v2 lock or recipe")
    if (independent["root_summary_sha256"]!=file_sha(folder/"root_summary.json")
        or independent["world_id"]!=world["world_id"]
        or not independent["native_900_checkpoint_byte_hashes_verified"]
        or not independent["sae_checkpoint_and_source_policy_hashes_verified"]
        or independent["confirmation_worlds_evaluated"]!=0
        or independent["reserved_contexts_evaluated"]!=0):
        raise AssertionError("original independent audit tampered")
    arms={arm:audit_arm(folder,world,source,arm) for arm in ("composed","additive")}
    primary=arms["composed"]["pairs"].get("topk/topk_alt")
    valid=(arms["composed"]["clean_correct"]/768>=.9 and primary is not None
           and primary["eligible_physical_signal"])
    reported=root["primary_effect"]
    expected_primary=primary["left_minus_right_correct"]/768 if valid else 0.
    if reported["planned_world_valid"]!=valid or abs(
            float(reported["root_primary_effect"])-expected_primary)>1e-10:
        raise AssertionError("v2 ITT root effect not consistent with raw results")
    additive=arms["additive"]["pairs"].get("topk/topk_alt")
    native_sham=arms["composed"]["pairs"].get("native/sham")
    valid_add=(arms["additive"]["clean_correct"]/768>=.9 and additive is not None
               and additive["eligible_physical_signal"])
    valid_native=(native_sham is not None and native_sham["eligible_physical_signal"])
    if (reported["additive_sham_valid"]!=valid_add
        or reported["native_sham_valid"]!=valid_native):
        raise AssertionError("sham availability differs from original")
    expected_shams=[
        additive["left_minus_right_correct"]/768 if valid_add else 1.,
        native_sham["left_minus_right_correct"]/768 if valid_native else 1.]
    if any(abs(float(x)-y)>1e-10 for x,y in zip(reported["sham_effects"],expected_shams)):
        raise AssertionError("unmeasured shams were imputed instead of conservatively bounded")
    if len(reported["sham_effects"])!=2:
        raise AssertionError("missing null family")
    return {
        "world_id":world["world_id"],"world_index":index,
        "truth_table_16":world["truth_table_16"],
        "composed_grammar":world["program"],
        "focal_positions":world["focal_positions"],
        "original_world_sha256":world["world_sha256"],
        "root_summary_sha256":file_sha(folder/"root_summary.json"),
        "source_manifest_sha256":source["sha256"],
        "clean_primary_admissible":valid,
        "root_ITT_correct_gain":round(expected_primary*768),
        "root_ITT_effect":expected_primary,
        "null_floor_fully_observed":valid_add and valid_native and valid,
        "failure_reasons":reported["failure_reasons"],
        "control_failures":reported["control_failures"],
        "arms":arms,
        "reserved_contexts_scored":0,
        "confirmation_worlds_scored":0}


def aggregate(root: Path, *, lock_path: Path, output: Path) -> dict:
    lock=read(lock_path)
    if (lock["schema"]!="NEC-03H/v2-fresh-development-lock/v1"
        or lock["new_development_world_indices"]!=list(INDICES)
        or lock["confirmation_worlds_evaluated"]!=0):
        raise AssertionError("NEC-03H original locked cohort changed")
    lock_sha=file_sha(lock_path)
    children=sorted(root.glob("nec03h-v2-development-root-*"))
    if (len(children)!=8 or
        set(int(x.name.removeprefix("nec03h-v2-development-root-")) for x in children)!=set(INDICES)):
        raise AssertionError("must use exactly the eight original development-root artifacts")
    by_index={}
    for child in children:
        i=int(child.name.removeprefix("nec03h-v2-development-root-"))
        by_index[i]=audit_root(child,i,lock_sha)
    ordered=[by_index[i] for i in INDICES]
    effects=[x["root_ITT_effect"] for x in ordered]
    roots_leave_one_out=[
        {"excluded":x["world_id"],"mean_remaining":mean(
            y["root_ITT_effect"] for y in ordered if y["world_id"]!=x["world_id"])}
        for x in ordered]
    report={
        "schema":SCHEMA,
        "root_unit":"distinct truth function/focal-position quartet with freshly seeded composed and additive model; all within-root contexts correlated",
        "root_indices":list(INDICES),
        "total_roots":8,
        "valid_primary_roots":sum(x["clean_primary_admissible"] for x in ordered),
        "roots_with_all_shams_feasible":sum(x["null_floor_fully_observed"] for x in ordered),
        "ITT_mean":mean(effects),
        "ITT_signed_prediction_gain_sum":sum(x["root_ITT_correct_gain"] for x in ordered),
        "root_ITT_effects":effects,
        "root_ITT_positive_negative_zero":{
            "positive":sum(e>0 for e in effects),
            "negative":sum(e<0 for e in effects),
            "zero":sum(e==0 for e in effects)},
        "sensitivity_leave_one_world_out":roots_leave_one_out,
        "sham_95_95_tail_calibration":"requires >=59 valid independent sham worlds; only 2 available with full sham set",
        "admission_registrations_unchanged":True,
        "original_v2_lock_sha256":lock_sha,
        "original_registry_sha256":registry()["registry_sha256"],
        "original_confirmation_world_commitment_sha256":
            registry()["confirmation_full_spec_sha256"],
        "independent_source_artifact_verifications_passed":True,
        "reserved_contexts_scored":0,
        "confirmation_worlds_scored":0,
        "worlds":ordered,
        "statistical_status":"posthoc decomposition only; no world-level significance claim",
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    output.mkdir(parents=True,exist_ok=True)
    (output/"nec03i_forensic_context_and_dose.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    summary={
        "schema":SCHEMA,
        "evidence_class":"posthoc read-only diagnosis",
        "ITT_mean":report["ITT_mean"],
        "valid_primary_roots":report["valid_primary_roots"],
        "roots_with_all_shams_feasible":report["roots_with_all_shams_feasible"],
        "confirmation_worlds_scored":0,
        "rows":[{
            "world_id":w["world_id"],
            "truth_table_16":hex(w["truth_table_16"]),
            "focal_positions":w["focal_positions"],
            "model_site_kind":w["arms"]["composed"]["site_kind"],
            "clean_correct_of_768":w["arms"]["composed"]["clean_correct"],
            "admissible":w["clean_primary_admissible"],
            "ITT_correct_gain":w["root_ITT_correct_gain"],
            "raw_topk_correct_gain":w["arms"]["composed"]["pairs"].get(
                "topk/topk_alt",{}).get("left_minus_right_correct"),
            "primary_signal":w["arms"]["composed"]["pairs"].get(
                "topk/topk_alt",{}).get("eligible_physical_signal"),
            "source_alternative_wins_cases":w["arms"]["composed"]["pairs"].get(
                "topk/topk_alt",{}).get("right_only_correct_cases"),
            "top_source_wins_cases":w["arms"]["composed"]["pairs"].get(
                "topk/topk_alt",{}).get("left_only_correct_cases"),
            "full_shams_feasible":w["null_floor_fully_observed"],
        } for w in ordered]}
    (output/"nec03i_compact_findings.json").write_text(json.dumps(summary,indent=2)+"\n")
    return report


def main(argv=None) -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots",type=Path,required=True)
    ap.add_argument("--lock",type=Path,default=Path(__file__).with_name("NEC03H_V2_LOCK.json"))
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args(argv)
    result=aggregate(args.roots,lock_path=args.lock,output=args.out)
    print(json.dumps({"status":result["statistical_status"],
                      "roots":result["total_roots"],
                      "eligible":result["valid_primary_roots"],
                      "ITT_mean":result["ITT_mean"],
                      "matched_shams":result["roots_with_all_shams_feasible"],
                      "confirmation_worlds_scored":0},sort_keys=True))


if __name__=="__main__":
    main()
