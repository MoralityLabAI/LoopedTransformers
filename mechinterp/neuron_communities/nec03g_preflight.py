"""NEC-03G: join independent development worlds for EvalMath FSI preflight.

Only development-world receipts, one paired effect per distinct world and
matched sham summaries are admitted. The confirmation cohort is represented
by SHA256 commitments, without any trained checkpoint, model score or raw
readout. Later decisions must use the pinned EvalMath power preflight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechinterp.neuron_communities.nec03g_worlds import (
    DEV_COUNT, CONFIRM_COUNT, encoded, registry, worlds,
)
import hashlib


def sha_file(path: Path) -> str:
    """Pure stdlib hash of a source-locked receipt, no Torch dependency."""
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

PILOT_COUNT=8


def aggregate(root: Path, output: Path) -> dict:
    manifests=list(root.rglob("root_summary.json"))
    if len(manifests)!=PILOT_COUNT:
        raise AssertionError(
            f"expected exactly {PILOT_COUNT} independent root summaries; found {len(manifests)}")
    expected=registry()
    permitted={w["world_id"]:w for w in worlds()[:PILOT_COUNT]}
    by_id={}
    for path in manifests:
        row=json.loads(path.read_text())
        world_id=row["world_id"]
        if world_id not in permitted or world_id in by_id:
            raise AssertionError("unexpected or repeated development world in pilot")
        if (row["cohort"]!="development"
            or row["registry_sha256"]!=expected["registry_sha256"]
            or row["world_sha256"]!=permitted[world_id]["world_sha256"]):
            raise AssertionError("development world differs from frozen registry")
        if not row["full_registered_development_recipe"]:
            raise AssertionError("short engineering run cannot supply EvalMath pilot")
        if row["original_confirmation_worlds_evaluated"]!=0 or row["world_reserved_context_groups_evaluated"]!=0:
            raise AssertionError("confirmation or reserved outcome opened")
        audit_path=path.parent/"independent_source_audit.json"
        independent=json.loads(audit_path.read_text())
        if (independent["root_summary_sha256"]!=sha_file(path)
            or independent["world_id"]!=world_id
            or not independent["confirmation_cohort_still_unopened"]):
            raise AssertionError("independent source/world audit failed")
        by_id[world_id]=(row,independent)
    ordered=[by_id[w["world_id"]][0] for w in worlds()[:PILOT_COUNT]]
    pilot=[]
    for row in ordered:
        e=row["primary_effect"]
        pilot.append({"unit_id":row["world_id"],
                      "paired_effect":str(e["root_primary_effect"]),
                      "sham_effects":[str(z) for z in e["sham_effects"]]})
    design={
        "schema_version":1,
        "experiment_id":"NEC-03G/FRESH-128-BOOLEAN-WORLD-PILOT-20261121",
        "independent_unit":"one independently initialized and trained two-teacher model pair per distinct Boolean truth-table and four-position world; repeated contexts stay within root",
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
        "pilot":pilot,
    }
    output.mkdir(parents=True,exist_ok=True)
    (output/"evalmath_development_pilot.json").write_bytes(encoded(design))
    result={
        "schema":"NEC-03G/independent-root-aggregation-v1",
        "evidence_class":"development-pilot-only",
        "registry_sha256":expected["registry_sha256"],
        "confirmation_commitments":len(expected["confirmation_commitments"]),
        "confirmation_worlds_trained":0,
        "confirmation_worlds_scored":0,
        "worlds_with_complete_source_receipts":PILOT_COUNT,
        "development_world_indices":list(range(PILOT_COUNT)),
        "primary_mean_effect":sum(float(x["paired_effect"]) for x in pilot)/PILOT_COUNT,
        "worlds_meeting_all_eligibility_gates":sum(
            r["primary_effect"]["planned_world_valid"] for r in ordered),
        "worlds_failing_gates":[
            {"world_id":row["world_id"],
             "reasons":row["primary_effect"]["failure_reasons"]}
            for row in ordered if not row["primary_effect"]["planned_world_valid"]],
        "paired_effects":pilot,
        "external_sham_quantile_calibrated":False,
        "independent_world_population_validated_for_inference":False,
        "confirmatory_test_authorized":False,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "note":"The 128 worlds are structurally distinct, but each pair shares a generator/architecture. Need >=59 independent development sham roots for one-sided 95/95 upper tolerance floor; stop before confirmation."
    }
    (output/"pilot_aggregation.json").write_bytes(encoded(result))
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(argv)
    result=aggregate(a.root,a.out)
    print(json.dumps({"development_worlds_audited":result["worlds_with_complete_source_receipts"],
                      "worlds_eligible":result["worlds_meeting_all_eligibility_gates"],
                      "pilot_mean_effect":result["primary_mean_effect"],
                      "confirmation_worlds_scored":0,
                      "preflight_authorized":False},sort_keys=True))


if __name__=="__main__":
    main()
