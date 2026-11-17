"""NW-01 read-only first original model/activation/checkpoint byte audit.

Revalidates the original 3-seed CPU model pilot from hosted run #44145535924.
The checker never fits a model, regenerates a source, selects interventions or
modifies original science PASS/FAIL. Uses only the Python standard library.
"""
from __future__ import annotations
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
RESULTS=HERE/"results"
REPORT_SHA="bf1fc415802df28def64e8f5a2a16a3933d78d2f11ada0df5f0976569baf227d"
ZIP_SHA="f2eb4383ca540d41ebdd235504df8e544a96f3823988148edc3babf8fec09545"
FIRST_REGISTRATION="42983fba14f3a616e9697523bce8a533fa8c6a84"
ORIGINAL_NOR42_SOURCE="f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"
ORIGINAL_NOR42_GRADE="bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae"
SOURCE_SHA={
    "data.py":"d787c2e14edd73ccae515ce9cc491017cf8357a795ae2c0249c8246aef8de551",
    "model.py":"a3a6a7ff5577d8acf4458dbfbbb655a3b69adedc04c07b3eff9fb51ea9676711",
    "run.py":"b2194451ff50eae6109f6353c6d9079014cb8acb4ad2f6a56ad7cbf005a8167f"
}
FIVE_GATES=(
    "authenticated_witness_readout_reachability",
    "full_state_donor_specificity_vs_equal_norm_sham",
    "paired_public_and_sham_information_firewall",
    "recurrent_witness_state_retention",
    "source_model_and_evalmath_root_integrity",
)


def sha(raw):
    return sha256(raw).hexdigest()


def audit():
    manifest=json.loads((RESULTS/"nw01_original_first_frozen_receipt.json").read_bytes())
    assert manifest["schema"]=="nw01.original_three_seed_recurrent_pilot_archive.v1"
    assert manifest["first_registration_commit"]==FIRST_REGISTRATION
    assert manifest["first_hosted_run"]==44145535924
    assert manifest["first_artifact_id"]==14357125920
    assert manifest["original_report_bytes"]==162716
    assert manifest["original_report_sha256"]==REPORT_SHA
    assert manifest["original_source_sha256"]==ORIGINAL_NOR42_SOURCE
    assert manifest["original_grade_sha256"]==ORIGINAL_NOR42_GRADE
    assert manifest["original_science_pass"] is True
    assert manifest["source_root_units"]==64
    assert manifest["source_generator_blocks"]==16
    assert manifest["train_synthetic_roots"]==512
    assert manifest["model_training_seeds"]==[11,23,47]
    assert manifest["original_source_code_sha256"]==SOURCE_SHA
    assert set(manifest["original_five_gates"])==set(FIVE_GATES)
    assert all(manifest["original_five_gates"].values())
    for name,expected in SOURCE_SHA.items():
        actual=sha((HERE/name).read_bytes())
        assert actual==expected,(name,actual,expected)
    raw=(RESULTS/"nw01_original_first_hosted.zip").read_bytes()
    assert len(raw)==manifest["archive_zip_bytes"]==1354690
    assert sha(raw)==manifest["archive_zip_sha256"]==ZIP_SHA
    with ZipFile(RESULTS/"nw01_original_first_hosted.zip") as z:
        assert z.testzip() is None
        names=sorted(z.namelist())
        assert len(names)==7
        assert names==sorted(manifest["original_files_sha256"])
        original={}
        for name in names:
            buf=z.read(name)
            wanted=manifest["original_files_sha256"][name]
            assert len(buf)==wanted["bytes"]
            assert sha(buf)==wanted["sha256"]
            original[name]=buf
    report=original["nw01_original_pilot.json"]
    assert len(report)==162716 and sha(report)==REPORT_SHA
    data=json.loads(report)
    assert data["schema"]=="nw01.original_three_seed_cpu_paired_witness_v1"
    assert data["registration_first_commit"]==FIRST_REGISTRATION
    assert data["source_first_sha256"]==ORIGINAL_NOR42_SOURCE
    assert data["first_grade_sha256"]==ORIGINAL_NOR42_GRADE
    assert data["public_only_feature_roster_sha256"]==(
        manifest["original_public_feature_manifest_sha256"])
    assert data["model_source_file_sha256"]==SOURCE_SHA
    assert data["source_units"]==64 and data["replicate_blocks"]==16
    assert data["paired_true_world_rows"]==128
    assert data["synthetic_train_roots"]==512
    assert data["training_seeds"]==[11,23,47]
    assert data["train_updates_per_seed"]==200
    assert data["registered_five_gates"]==manifest["original_five_gates"]
    assert data["pilot_pass"] is True
    assert len(data["by_seed"])==3
    for i,s in enumerate(data["by_seed"]):
        assert s["training_seed"]==[11,23,47][i]
        assert len(s["per_root"])==64
        assert len({r["root_id"] for r in s["per_root"]})==64
        assert {r["replicate"] for r in s["per_root"]}==set(range(100,116))
        assert all(sum(r["replicate"]==rep for r in s["per_root"])==4
                   for rep in range(100,116))
        assert s["depth_outcomes"]["public"]["8"]["accuracy"]==.5
        assert s["depth_outcomes"]["sham"]["8"]["accuracy"]==.5
        assert s["depth_outcomes"]["authenticated"]["8"]["accuracy"]==1.0
        assert s["depth_outcomes"]["public"]["8"]["brier"]>=.25
        assert s["depth_outcomes"]["sham"]["8"]["brier"]>=.25
        assert s["causal_donor_swaps"]["4"]["mean_donor_minus_random"]>=.10
        assert s["raw_activation_sha256"]==sha(original[s["raw_activation_file"]])
        assert s["checkpoint_sha256"]==sha(original[s["checkpoint_file"]])
        for key in ("public","sham"):
            assert s["matched_pair_numeric_firewall"][key]["max_readout_logit_difference"]<=1e-7
            assert s["matched_pair_numeric_firewall"][key]["max_recurrent_state_difference"]<=1e-7
        for effect in s["causal_donor_swaps"].values():
            assert effect["max_donor_trace_error"]<1e-5
            assert effect["max_same_world_identity_error"]<1e-6
    return {
        "schema":"nw01.immutable_first_original_model_audit.v1",
        "first_run":44145535924,
        "pilot_pass":True,
        "five_gates":manifest["original_five_gates"],
        "source_units":64,"replicate_blocks":16,
        "trace_files":3,"checkpoints":3,
        "original_model_report_sha256":REPORT_SHA,
        "original_complete_zip_sha256":ZIP_SHA
    }


if __name__=="__main__":
    print(json.dumps(audit(),sort_keys=True,indent=2))
