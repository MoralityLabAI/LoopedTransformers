"""NEC-03H: collect exactly eight new dev-008..015 root audits for FSI-01.

This reader accepts only already completed, fully trained, source-audited
fresh development worlds. The 64 confirmation worlds and within-world
reserved 48-context splits remain sealed. Missing sham controls receive
conservative unit effect bounds of 1.0 in the preflight; they are never
silently treated as zero-effect control worlds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib

from mechinterp.neuron_communities.nec03g_worlds import (
    encoded, worlds, registry, CONFIRM_COUNT,
)

SCHEMA="NEC-03H/source-site-fallback-900steps-development-v2"
INDICES=tuple(range(8,16))
EXPECTED=tuple(f"dev-{i:03d}" for i in INDICES)


def sha_file(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()


def aggregate(root:Path,out:Path):
    paths=list(root.rglob("root_summary.json"))
    if len(paths)!=8:
        raise AssertionError(f"expected eight fresh development worlds, found {len(paths)}")
    names=set()
    outcomes={}
    for p in paths:
        row=json.loads(p.read_text())
        index=row["world_index"]
        if index not in INDICES:
            raise AssertionError("cannot mix opened/confirmation root into NEC03H-v2")
        trusted=worlds()[index]
        name=trusted["world_id"]
        if name in names or row["world_id"]!=name or row["world_sha256"]!=trusted["world_sha256"]:
            raise AssertionError("duplicated or unregistered fresh development world")
        names.add(name)
        lock_path=Path(__file__).with_name("NEC03H_V2_LOCK.json")
        if row.get("v2_protocol_lock_sha256")!=sha_file(lock_path):
            raise AssertionError("root was not run under the immutable v2 protocol")
        if (row["schema"]!=SCHEMA
            or row["original_NEC03G_world_registry_sha256"]!=registry()["registry_sha256"]
            or row["cohort"]!="development"
            or not row["full_registered_v2_recipe"]
            or row["trunk_steps"]!=900 or row["sae_steps"]!=480
            or row["reserved_context_groups_evaluated"]!=0
            or row["original_confirmation_worlds_evaluated"]!=0):
            raise AssertionError("unfrozen development recipe or leaked cohort")
        audited=json.loads((p.parent/"independent_source_audit.json").read_text())
        if (audited["root_summary_sha256"]!=sha_file(p)
            or audited["world_id"]!=name
            or not audited["native_900_checkpoint_byte_hashes_verified"]
            or not audited["sae_checkpoint_and_source_policy_hashes_verified"]
            or not audited["all_local_physical_doses_and_active_feature_ranks_recomputed"]
            or audited["confirmation_worlds_evaluated"]!=0
            or audited["reserved_contexts_evaluated"]!=0):
            raise AssertionError("new-world independent source/intervention audit absent")
        if (str(row["primary_effect"]["root_primary_effect"])!=audited["primary_effect"]
            or row["primary_effect"]["planned_world_valid"]!=audited["primary_eligible"]):
            raise AssertionError("root ITT effect differs from independent audit")
        outcomes[name]=row
    if names!=set(EXPECTED):
        raise AssertionError("incomplete frozen v2 world slots")
    rows=[outcomes[name] for name in EXPECTED]
    pilot=[]
    for r in rows:
        e=r["primary_effect"]
        if len(e["sham_effects"])!=2:
            raise AssertionError("expected additive and native-sham control effect floors")
        pilot.append({"unit_id":r["world_id"],
                      "paired_effect":str(e["root_primary_effect"]),
                      "sham_effects":[str(x) for x in e["sham_effects"]]})
    spec={
        "schema_version":1,
        "experiment_id":"NEC-03H-V2/FRESH-DEVELOPMENT-008-015",
        "independent_unit":"one unique Boolean truth table/position quartet with independently seeded composed and additive 900-step PEF-07 checkpoints; 48 context quartets nested within root",
        "alpha":"0.05",
        "scientific_min_effect":"0.02",
        "target_effect":"0.04",
        "measurement_resolution":str(1/768),
        "sham_quantile":"0.95",
        "sham_confidence":"0.95",
        "planned_comparisons":3,
        "confirm_units_planned":CONFIRM_COUNT,
        "pilot_from_development_only":True,
        "heldout_sealed":True,
        "units_independent":False,
        "pilot":pilot
    }
    report={
        "schema":SCHEMA,"phase":"eight-fresh-world-development-root-pilot",
        "worlds":list(EXPECTED),
        "registered_world_population_sha256":registry()["registry_sha256"],
        "root_effects":pilot,
        "worlds_development_audited":8,
        "worlds_primary_eligible":sum(r["primary_effect"]["planned_world_valid"] for r in rows),
        "worlds_with_all_sham_controls_eligible":sum(
            r["primary_effect"]["null_floor_calibration_eligible"] for r in rows),
        "source_fallback_use_by_arm":{
            arm:sum(r["source_site_kinds"][arm]=="fallback" for r in rows)
            for arm in ("composed","additive")},
        "clean_competence_by_arm":{
            arm:sum(r["native_clean_scores"][arm]["accuracy"]>=.90 for r in rows)
            for arm in ("composed","additive")},
        "worlds_primary_failed":[
            {"world_id":r["world_id"],
             "reasons":r["primary_effect"]["failure_reasons"],
             "control_failures":r["primary_effect"]["control_failures"]}
            for r in rows if not r["primary_effect"]["planned_world_valid"]],
        "mean_intention_to_treat_effect":sum(float(r["primary_effect"]["root_primary_effect"]) for r in rows)/8,
        "sham_floor_is_not_calibrated":True,
        "conservative_bound_for_missing_shams":1.0,
        "confirmation_worlds_models_or_outcomes_exposed":0,
        "reserved_contexts_exposed":0,
        "frozen_confirmation_spec_commitment":registry()["confirmation_full_spec_sha256"],
        "permission_to_begin_confirmation":False,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "note":"Eight development roots cannot resolve 95/95 upper sham tolerance. Exact once-root effects and conservative missing-sham bounds are passed to pinned EvalMath."
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"evalmath_v2_pilot.json").write_bytes(encoded(spec))
    (out/"nec03h_v2_aggregate.json").write_bytes(encoded(report))
    print(json.dumps({"worlds":8,
                      "primary_eligible":report["worlds_primary_eligible"],
                      "sham_eligible":report["worlds_with_all_sham_controls_eligible"],
                      "mean_itt":report["mean_intention_to_treat_effect"],
                      "confirmation_worlds_opened":0},sort_keys=True))
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    aggregate(args.root,args.out)


if __name__=="__main__":
    main()
