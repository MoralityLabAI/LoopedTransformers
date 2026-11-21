"""NEC-03N four-root source-only 900 vs 1800 fixed competence aggregate.

Require four independently audited, exactly milestone-replayed native
teacher pairs. The source/split world is the unit; all 2560 histories in
each world are in-sample for these models. Fails if any forbidden cohort
or old four-input confirmation has been opened.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

from mechinterp.neuron_communities.nec03l_five_input_worlds import registration,worlds
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry

SCHEMA="NEC-03N/four-root-milestone-native-learning-aggregate/v1"
INDICES=(0,1,2,3)


def digest_file(path:Path):
    h=sha256()
    with path.open("rb") as f:
        for blob in iter(lambda:f.read(1<<20),b""):
            h.update(blob)
    return h.hexdigest()


def aggregate(folder:Path,out:Path,lock:Path):
    inputs=list(folder.glob("nec03n-source-world-*"))
    if len(inputs)!=4 or {int(x.name.rsplit("-",1)[1]) for x in inputs}!=set(INDICES):
        raise AssertionError("need four complete original source-only calibration roots")
    rows=[]
    for idx in INDICES:
        root=folder/f"nec03n-source-world-{idx}"
        row=json.loads((root/"native1800_source_calibration.json").read_text())
        audit=json.loads((root/"independent_milestone_audit.json").read_text())
        w=worlds()[idx]
        if (row["schema"]!="NEC-03N/source-only-fixed1800-continuous-optimizer-v1"
            or row["world_id"]!=w["world_id"]
            or row["world_sha256"]!=w["world_sha256"]
            or row["new_1800_source_lock_sha256"]!=digest_file(lock)
            or row["new_five_input_registry_sha256"]!=registration()["registry_sha256"]
            or row["old_original_four_input_confirmation_sha256"]!=
                old_registry()["confirmation_full_spec_sha256"]
            or row["native_1800_optimizer_steps_per_arm"]!=1800
            or row["new_dev_evaluation_groups_scored"]!=0
            or row["reserved_groups_scored"]!=0
            or row["embargoed_worlds_scored"]!=0
            or row["old_confirmation_worlds_scored"]!=0
            or row["new_source_AE_models_fitted"]!=0):
            raise AssertionError("wrong source/calibration scientific cohort")
        if (audit["new_source_calibration_sha256"]!=digest_file(root/"native1800_source_calibration.json")
            or audit["world_id"]!=w["world_id"]
            or not audit["original_900_models_identical_at_milestone"]
            or not audit["all_2560_source_training_input_predictions_recomputed"]
            or not audit["native_1800_checkpoints_and_weight_digests_verified"]
            or audit["new_development_evaluation_groups_scored"]!=0
            or audit["embargoed_worlds_scored"]!=0
            or audit["old_confirmation_worlds_scored"]!=0):
            raise AssertionError("missing exact 900 source trainer or 1800 independently audited evidence")
        arms={}
        for a in ("composed","additive"):
            entry=audit["arms"][a]
            if (entry["source_correct_step900"] not in range(2561)
                or entry["source_correct_step1800"] not in range(2561)
                or not entry["frozen_900_optimizer_stream_replayed_exactly"]):
                raise AssertionError("wrong original native trainer replay")
            arms[a]=entry
        rows.append({
            "world_id":w["world_id"],
            "world_index":idx,
            "original_source_world_sha256":w["world_sha256"],
            "new_native_source_900_to_1800":arms,
            "new_source_1800_receipt_sha256":audit["new_source_calibration_sha256"],
        })
    models=[r["new_native_source_900_to_1800"][arm]
            for r in rows for arm in ("composed","additive")]
    def eligible_arms(key):
        return sum(x[key] for x in models)
    def eligible_worlds(key):
        return sum(all(r["new_native_source_900_to_1800"][a][key]
                       for a in ("composed","additive")) for r in rows)
    result={
        "schema":SCHEMA,
        "registered_old_source_roots":4,
        "native_model_comparisons":8,
        "source_teacher_training_steps_original":900,
        "source_teacher_training_steps_new":1800,
        "original_900_model_parameters_reproduced_exactly":True,
        "original_900_source_competent_models":eligible_arms("native_competent_step900"),
        "new_1800_source_competent_models":eligible_arms("native_competent_step1800"),
        "original_worlds_both_teachers_source_competent":eligible_worlds("native_competent_step900"),
        "new_worlds_both_teachers_source_competent":eligible_worlds("native_competent_step1800"),
        "all_source_training_histories_scored_per_teacher":2560,
        "new_five_input_registry_sha256":registration()["registry_sha256"],
        "original_four_input_confirmation_spec_sha256":
            old_registry()["confirmation_full_spec_sha256"],
        "original_900_results_preserved":True,
        "new_five_input_development_evaluation_contexts_scored":0,
        "reserved_contexts_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "new_SAE_or_causal_eval_executed":False,
        "inference_scope":"paired in-sample source-only learning schedule calibration",
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "worlds":rows,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec03n_native900_vs_1800_source_accuracy.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NEC03N_EXACT_900_TO_1800_FOUR_WORLD_SOURCE_COMPARISON_AUDITED")
    return result


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lock",type=Path,default=Path(__file__).with_name("NEC03N_SOURCE_LOCK.json"))
    a=p.parse_args(argv)
    aggregate(a.root,a.out,a.lock)


if __name__=="__main__":
    main()
