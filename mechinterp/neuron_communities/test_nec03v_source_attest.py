"""Metadata-only provenance tests, using fabricated archive BYTES for QA.

Tests cannot make the scientific attestation entrypoint trust these fixtures:
they call the internal source validator with a synthetic root lock and retain
original_archival_lock_verified=False.
"""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mechinterp.neuron_communities.nec03v_source_attest import (
    SCHEMA, attest, canonical, sha256_file, verify_archive,
)


def file(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_file(path)


def make(root: Path, index: int):
    source, t, reset = root/"source", root/"nec03t", root/"reset"
    source.mkdir()
    t.mkdir()
    reset.mkdir()
    world_id = f"dev5-{index:03d}"
    world_sha = hashlib.sha256(world_id.encode()).hexdigest()
    native_model_sha = {}
    sae_sha = {}
    entries = {}
    for arm in ("composed","additive"):
        native_file = f"{arm}.reset1800.pt"
        native_dir = reset if index < 4 else source
        native_sha = file(native_dir/native_file, f"SYNTHETIC-NATIVE-{index}-{arm}".encode())
        model_sha = hashlib.sha256(f"SYNTHETIC-MODEL-{index}-{arm}".encode()).hexdigest()
        native_model_sha[arm] = model_sha
        ae_file = f"{arm}.sae.pt"
        sae_sha[arm] = file(source/ae_file, f"SYNTHETIC-SAE-{index}-{arm}".encode())
        roster_sha = file(source/f"{arm}.source_roster.json",
                          canonical({"source_sites": [1, 2], "source_hubs": []}).encode())
        fit = {
            "source_dictionary_file": ae_file,
            "source_dictionary_sha256": sae_sha[arm],
            "source_contexts_fit": 40,
            "source_rank_context_groups": 40,
            "sae_features": 128,
            "active_codes_target": 16,
        }
        native = (
            {"new_checkpoint_file": native_file,
             "new_checkpoint_sha256": native_sha,
             "new_model_sha256": model_sha} if index < 4 else
            {"reset1800_checkpoint_file": native_file,
             "reset1800_checkpoint_sha256": native_sha,
             "reset1800_model_sha256": model_sha}
        )
        entries[arm] = {
            "source_roster_sha256": roster_sha,
            "source_candidate": {"nodes": [1, 2]},
            "SAE_source": fit,
            "native_reset1800" if index < 4 else "native": native,
            "source_model_eligible": True,
        }
    sname = ("nec03o_source_refit_receipt.json" if index < 4 else
             "nec03ob_source_receipt.json")
    source_receipt = {
        "world_id": world_id, "world_index": index, "world_sha256": world_sha,
        "source_manifest_sha256": "b"*64, "source_fit_groups": 40,
        "source_rank_groups": 40,
        "both_arms_source_eligible": True,
        "development_evaluation_groups_scored": 0,
        "reserved_context_groups_scored": 0, "embargoed_worlds_scored": 0,
        "four_input_confirmation_worlds_scored": 0,
        "arms": entries,
    }
    source_sha=file(source/sname, canonical(source_receipt).encode())
    aname=("independent_nec03o_source_audit.json" if index < 4 else
           "independent_nec03ob_source_audit.json")
    audit_key = ("source_refit_receipt_sha256" if index < 4 else
                 "source_receipt_sha256")
    audit_pass=("fresh_1800_specific_source_SAE_and_rank_audited" if index < 4
                else "both_arm_source_SAEs_and_geometry_independently_checked")
    file(source/aname, canonical({
        "world_sha256": world_sha, audit_key: source_sha,
        audit_pass: True,
    }).encode())
    original_lock = sha256_file(
        Path(__file__).with_name("NEC03T_SOURCE_LOCK.json")
    )
    t_arms = {}
    for arm in ("composed","additive"):
        raw_name=f"{arm}.nec03t_source_necessity.jsonl"
        raw_sha=file(t/raw_name, b"SYNTHETIC_OLD_CASE_ARM_DIGEST\n")
        t_arms[arm] = {
            "source_model_sha256": native_model_sha[arm],
            "source_sae_checkpoint_sha256": sae_sha[arm],
            "source_receipt_sha256": source_sha,
            "source_groups_scored": list(range(24, 32)),
            "source_histories": 256,
            "original_arm_source_eligible": True,
            "raw_file":raw_name,"raw_sha256":raw_sha,
        }
    root_report = {
        "schema": "NEC-03T/source-only-full-affine-necessity-ablation/v1",
        "world_index": index, "world_id": world_id, "world_sha256": world_sha,
        "source_lock_sha256": original_lock,
        "source_manifest_sha256": "b"*64,
        "original_ITT_source_failure": False,
        "source_groups_scored": list(range(24,32)),
        "source_training_cases_per_arm":256,
        "arms":t_arms,
    }
    t_sha=file(t/"nec03t_source_necessity_report.json",canonical(root_report).encode())
    audit_sha=file(t/"independent_nec03t_source_necessity_audit.json",b"SYNTHETIC audit bytes")
    lock={"roots":{str(index):{
        "report_sha256":t_sha,"audit_sha256":audit_sha,
        "world_sha256":world_sha,
    }}}
    return source,t,reset,lock,original_lock


class SourceOriginTests(unittest.TestCase):
    def test_both_original_archive_shapes_accept_matching_synthetic_bytes(self):
        for index in (0,4):
            with tempfile.TemporaryDirectory() as temporary:
                source,t,reset,lock,old=make(Path(temporary),index)
                out=verify_archive(index,source,t,reset if index<4 else None,
                                   parent_lock=lock,
                                   original_source_lock_sha256=old,
                                   historical_lock_verified=False)
                self.assertEqual(out["schema"],SCHEMA)
                self.assertTrue(out["source_checkpoint_bytes_verified"])
                self.assertFalse(out["original_archival_lock_verified"])
                self.assertEqual(out["model_forwards"],0)
                self.assertFalse(out["projected_decoder_columns_verified"])
                self.assertEqual(set(out["source_checkpoint_receipts"]),
                                 {"composed","additive"})
                self.assertEqual(out["source_scope"],list(range(24,32)))

    def test_tampered_SAE_and_model_files_rejected(self):
        for index in (0,4):
            with tempfile.TemporaryDirectory() as temporary:
                source,t,reset,lock,old=make(Path(temporary),index)
                (source/"composed.sae.pt").write_bytes(b"MISMATCHED SAE")
                with self.assertRaisesRegex(ValueError,"archive bytes mismatch"):
                    verify_archive(index,source,t,reset if index<4 else None,
                                   parent_lock=lock,
                                   original_source_lock_sha256=old,
                                   historical_lock_verified=False)
            with tempfile.TemporaryDirectory() as temporary:
                source,t,reset,lock,old=make(Path(temporary),index)
                native_dir=reset if index<4 else source
                (native_dir/"additive.reset1800.pt").write_bytes(b"MISMATCHED NATIVE")
                with self.assertRaisesRegex(ValueError,"archive bytes mismatch"):
                    verify_archive(index,source,t,reset if index<4 else None,
                                   parent_lock=lock,
                                   original_source_lock_sha256=old,
                                   historical_lock_verified=False)

    def test_false_source_receipt_or_replaced_original_root_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,t,reset,lock,old=make(Path(temporary),0)
            p=t/"nec03t_source_necessity_report.json"
            p.write_bytes(p.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError,"historical archive bytes"):
                verify_archive(0,source,t,reset,parent_lock=lock,
                               original_source_lock_sha256=old,
                               historical_lock_verified=False)

    def test_renamed_unexpected_source_or_protected_world_denied(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,t,reset,lock,old=make(Path(temporary),4)
            p=source/"nec03ob_source_receipt.json"
            doc=json.loads(p.read_text())
            doc["new_development_evaluation_groups_scored"]=1
            p.write_text(canonical(doc))
            with self.assertRaises(ValueError):
                verify_archive(4,source,t,None,parent_lock=lock,
                               original_source_lock_sha256=old,
                               historical_lock_verified=False)
            with self.assertRaises(PermissionError):
                verify_archive(8,source,t,None,parent_lock=lock,
                               original_source_lock_sha256=old,
                               historical_lock_verified=False)

    def test_hardcoded_original_NECO3U_lock_rejects_fabricated_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,t,reset,lock,old=make(Path(temporary),0)
            with self.assertRaises((ValueError,KeyError)):
                attest(0,source,t,reset)

    def test_archive_path_must_remain_filename_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,t,reset,lock,old=make(Path(temporary),4)
            p=source/"nec03ob_source_receipt.json"
            doc=json.loads(p.read_text())
            doc["arms"]["composed"]["SAE_source"]["source_dictionary_file"]="../sae.pt"
            p.write_text(canonical(doc))
            tpath=t/"nec03t_source_necessity_report.json"
            record=json.loads(tpath.read_text())
            new_sha=sha256_file(p)
            for arm in ("composed","additive"):
                record["arms"][arm]["source_receipt_sha256"]=new_sha
            tpath.write_text(canonical(record))
            lock["roots"]["4"]["report_sha256"]=sha256_file(tpath)
            with self.assertRaises(ValueError):
                verify_archive(4,source,t,None,parent_lock=lock,
                               original_source_lock_sha256=old,
                               historical_lock_verified=False)


if __name__ == "__main__":
    unittest.main()
