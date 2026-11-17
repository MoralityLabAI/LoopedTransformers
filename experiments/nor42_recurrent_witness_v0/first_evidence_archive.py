"""NW-01 immutable first original three-seed CPU model evidence publisher.

Copies the complete original hosted activation and checkpoint receipts byte for
byte after the model has scored; does not fit, select, resample or modify any
source root. Retains a scientific FAIL if one originally occurred.
"""
from __future__ import annotations
import argparse,json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED,ZipInfo

FIRST_REGISTRATION="42983fba14f3a616e9697523bce8a533fa8c6a84"
FIRST_RUN_ID=44145535924
FIRST_ARTIFACT_ID=14357125920
FIRST_REPORT_BYTES=162716
FIRST_REPORT_SHA="bf1fc415802df28def64e8f5a2a16a3933d78d2f11ada0df5f0976569baf227d"
SOURCE_SHA="f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"
GRADE_SHA="bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae"
MODEL_SEEDS=(11,23,47)


def digest(x:bytes):
    return sha256(x).hexdigest()


def archive_original(folder:Path,output:Path):
    original=folder/"nw01_original_pilot.json"
    raw=original.read_bytes()
    if len(raw)!=FIRST_REPORT_BYTES or digest(raw)!=FIRST_REPORT_SHA:
        raise ValueError("first original NW01 report bytes differ")
    g=json.loads(raw)
    assert g["schema"]=="nw01.original_three_seed_cpu_paired_witness_v1"
    assert g["registration_first_commit"]==FIRST_REGISTRATION
    assert g["source_first_sha256"]==SOURCE_SHA
    assert g["first_grade_sha256"]==GRADE_SHA
    assert g["source_units"]==64 and g["replicate_blocks"]==16
    assert g["paired_true_world_rows"]==128
    assert tuple(g["training_seeds"])==MODEL_SEEDS
    assert g["synthetic_train_roots"]==512
    assert g["train_updates_per_seed"]==200
    assert g["recurrent_steps"]==8 and g["latent_dim"]==48
    assert len(g["by_seed"])==3
    assert len(g["registered_five_gates"])==5
    assert g["pilot_pass"]==all(g["registered_five_gates"].values())
    file_bytes={"nw01_original_pilot.json":raw}
    for entry,seed in zip(g["by_seed"],MODEL_SEEDS):
        assert entry["training_seed"]==seed
        assert len(entry["per_root"])==64
        assert {r["replicate"] for r in entry["per_root"]}==set(range(100,116))
        assert all(sum(r["replicate"]==block for r in entry["per_root"])==4
                   for block in range(100,116))
        for name,expected in (
            (entry["raw_activation_file"],entry["raw_activation_sha256"]),
            (entry["checkpoint_file"],entry["checkpoint_sha256"])
        ):
            buf=(folder/name).read_bytes()
            assert digest(buf)==expected
            file_bytes[name]=buf
    assert len(file_bytes)==7
    output.mkdir(parents=True,exist_ok=True)
    destination=output/"nw01_original_first_hosted.zip"
    with ZipFile(destination,"w",ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(file_bytes):
            info=ZipInfo(name,(2026,11,18,0,0,0))
            info.compress_type=ZIP_DEFLATED
            z.writestr(info,file_bytes[name],
                       compress_type=ZIP_DEFLATED,compresslevel=9)
    packed=destination.read_bytes()
    manifest={
        "schema":"nw01.original_three_seed_recurrent_pilot_archive.v1",
        "first_registration_commit":FIRST_REGISTRATION,
        "first_hosted_run":FIRST_RUN_ID,
        "first_artifact_id":FIRST_ARTIFACT_ID,
        "original_report_bytes":FIRST_REPORT_BYTES,
        "original_report_sha256":FIRST_REPORT_SHA,
        "original_source_sha256":SOURCE_SHA,
        "original_grade_sha256":GRADE_SHA,
        "original_science_pass":g["pilot_pass"],
        "original_five_gates":g["registered_five_gates"],
        "model_training_seeds":list(MODEL_SEEDS),
        "source_root_units":64,
        "source_generator_blocks":16,
        "train_synthetic_roots":512,
        "original_source_code_sha256":g["model_source_file_sha256"],
        "original_public_feature_manifest_sha256":
            g["public_only_feature_roster_sha256"],
        "original_files_sha256":{
            k:{"bytes":len(v),"sha256":digest(v)} for k,v in file_bytes.items()
        },
        "archive_zip_bytes":len(packed),
        "archive_zip_sha256":digest(packed),
        "evidence_level":"retrospective model pilot with previously published original NOR42 source pairs; all 3 model seeds and 16 source blocks",
        "external_witness_scope":"First discrepancy is synthetic NOR42 grade beta supplied to witnessed arm. Authenticity and public-only inferability are separate premises.",
    }
    dest=output/"nw01_original_first_frozen_receipt.json"
    dest.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("NW01 FIRST ORIGINAL MODEL REPORT BYTES SHA",len(raw),digest(raw))
    print("NW01 FIRST ORIGINAL COMPLETE ZIP BYTES SHA",
          len(packed),digest(packed))
    print("NW01 ORIGINAL PILOT PASS",g["pilot_pass"])
    print("NW01 ORIGINAL FIVE GATES",g["registered_five_gates"])
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    archive_original(a.original,a.output)


if __name__=="__main__":
    main()
