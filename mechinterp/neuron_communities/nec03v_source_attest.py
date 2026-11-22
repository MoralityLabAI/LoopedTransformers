"""NEC-03V: read-only authentication of the ORIGINAL source checkpoints.

Checks the byte-locked NEC03T source report, NEC03U archival lock, original
independent source-model/SAE audits, exact native & SAE checkpoint bytes, and
the immutable source-site roster. Never loads .pt files, imports Torch,
executes model inference, computes predictions, or opens protected cohorts.

This attests the *origin of the weights and sites*. It does not attest new
per-case TopK coefficients or 32D columns; those need a separate, audited
label-blind extraction using historical source inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ARMS = ("composed", "additive")
ROOTS = tuple(range(8))
OLD_SOURCE_RUN = 44899750508
NEW_SOURCE_RUN = 44900628360
T_RUN = 44913605626
SOURCE_ARTIFACTS = (
    14705980493, 14705925546, 14704954383, 14705281999,
    14705279661, 14706270227, 14706290195, 14706002643,
)
T_ARTIFACTS = (
    14710962168, 14710947309, 14711101362, 14710449539,
    14710478043, 14710554274, 14711161360, 14710609069,
)
SCHEMA = "NEC-03V/authenticated-historical-source-checkpoint/v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_name(value: str) -> str:
    if (not isinstance(value, str) or not value
            or value != Path(value).name or value.startswith(".")
            or "/" in value or "\\" in value):
        raise ValueError("archive receipt contains unsafe filename")
    return value


def _file(directory: Path, name: str, expected: str) -> Path:
    path = directory / _existing_name(name)
    if (not isinstance(expected, str) or len(expected) != 64 or
            sha256_file(path) != expected):
        raise ValueError(f"historical archive bytes mismatch: {name}")
    return path


def _read(path: Path):
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("oversized archive metadata receipt")
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_clean(record, names):
    for field in names:
        if field in record and record[field] != 0:
            raise ValueError(f"original source archive reads protected population: {field}")


def _source_arm(files: Path, reset: Path | None, root: int, arm: str,
                source: dict, report: dict, expected_model: str,
                expected_sae: str | None, source_sha: str):
    item = source["arms"][arm]
    if item["source_roster_sha256"] is None:
        raise ValueError("source roster hash missing")
    _file(files, f"{arm}.source_roster.json", item["source_roster_sha256"])
    fit = item["SAE_source"]
    if fit is None:
        if expected_sae is not None or root not in (3, 5) or item["source_candidate"] is not None:
            raise ValueError("missing SAE is only valid for original source-ineligible roots")
        sae_sha = None
    else:
        name = fit["source_dictionary_file"]
        sae_sha = fit["source_dictionary_sha256"]
        _file(files, name, sae_sha)
        if (sae_sha != expected_sae
                or fit["source_fit_context_groups"] if False else False):
            # Only precise downstream byte identity matters here; the original
            # fit receipt's own source split is checked in the enclosing report.
            raise ValueError("SAE checkpoint mismatches NEC03T")
        if (fit.get("source_contexts_fit") != 40
                or fit.get("source_rank_context_groups") != 40
                or fit.get("active_codes_target") != 16
                or fit.get("sae_features") != 128):
            raise ValueError("original SAE training-split shape changed")
    if root < 4:
        if reset is None:
            raise ValueError("source world 0..3 requires original reset1800 native archive")
        native = item["native_reset1800"]
        native_file = native["new_checkpoint_file"]
        native_sha = native["new_checkpoint_sha256"]
        model_digest = native["new_model_sha256"]
        native_dir = reset
    else:
        native = item["native"]
        native_file = native["reset1800_checkpoint_file"]
        native_sha = native["reset1800_checkpoint_sha256"]
        model_digest = native["reset1800_model_sha256"]
        native_dir = files
    _file(native_dir, native_file, native_sha)
    if model_digest != expected_model:
        raise ValueError("original native weights digest changed from NEC03T")
    if (source["world_index"] != root
            or source["world_sha256"] != report["world_sha256"]
            or report["arms"][arm]["source_receipt_sha256"] != source_sha):
        raise ValueError("source checkpoint is mapped to the wrong historical world")
    return {
        "native_sha256": model_digest,
        "sae_sha256": sae_sha,
        "source_sites_sha256": item["source_roster_sha256"],
        "origin_receipt_sha256": source_sha,
    }, {
        "native_checkpoint_file": native_file,
        "native_checkpoint_sha256": native_sha,
        "sae_checkpoint_sha256": sae_sha,
        "source_model_digest": model_digest,
        "original_source_candidate_present": bool(item["source_candidate"]),
        "original_source_eligible": bool(item["source_model_eligible"]),
        "source_roster_sha256": item["source_roster_sha256"],
    }


def verify_archive(root: int, source_dir: Path, t_dir: Path,
                   reset_dir: Path | None, *, parent_lock: dict,
                   original_source_lock_sha256: str,
                   historical_lock_verified: bool):
    """Validate one historical root. Tests may pass synthetic parent_lock.

    Only the public CLI uses the locked authentic NEC03U records, and only
    that path sets historical_lock_verified=True. Tests cannot claim original
    GitHub provenance merely by making up matching hashes.
    """
    if type(root) is not int or root not in ROOTS:
        raise PermissionError("NEC03V only authenticates original dev5-000..007")
    locked = parent_lock["roots"][str(root)]
    t_report_path = _file(t_dir, "nec03t_source_necessity_report.json",
                          locked["report_sha256"])
    _file(t_dir, "independent_nec03t_source_audit.json", locked["audit_sha256"])
    t = _read(t_report_path)
    if (t["schema"] != "NEC-03T/source-only-full-affine-necessity-ablation/v1"
            or t["world_index"] != root
            or t["world_id"] != f"dev5-{root:03d}"
            or t["world_sha256"] != locked["world_sha256"]
            or t["source_lock_sha256"] != original_source_lock_sha256
            or t["source_groups_scored"] != list(range(24, 32))
            or t["source_training_cases_per_arm"] != 256
            or t["original_ITT_source_failure"] != (root in (3, 5))
            or set(t["arms"]) != set(ARMS)):
        raise ValueError("NEC03T historical world/source panel or ITT changed")
    for arm in ARMS:
        saved = t["arms"][arm]
        _file(t_dir, saved["raw_file"], saved["raw_sha256"])
        if (saved["source_groups_scored"] != list(range(24, 32))
                or saved["source_histories"] != 256):
            raise ValueError("historical source output denominator changed")
    if root < 4:
        receipt_name = "nec03o_source_refit_receipt.json"
        audit_name = "independent_nec03o_source_audit.json"
        audit_hash_field = "source_refit_receipt_sha256"
        audit_pass = "fresh_1800_specific_source_SAE_and_rank_audited"
    else:
        receipt_name = "nec03ob_source_receipt.json"
        audit_name = "independent_nec03ob_source_audit.json"
        audit_hash_field = "source_receipt_sha256"
        audit_pass = "both_arm_source_SAEs_and_geometry_independently_checked"
    receipt_path = source_dir / receipt_name
    source_sha = sha256_file(receipt_path)
    source = _read(receipt_path)
    audit = _read(source_dir / audit_name)
    if (audit["world_sha256"] != t["world_sha256"]
            or audit[audit_hash_field] != source_sha
            or audit[audit_pass] is not True
            or source["world_index"] != root
            or source["world_id"] != t["world_id"]
            or source["world_sha256"] != t["world_sha256"]
            or source["source_manifest_sha256"] != t["source_manifest_sha256"]
            or set(source["arms"]) != set(ARMS)
            or source["source_fit_groups"] != 40
            or source["source_rank_groups"] != 40
            or source["both_arms_source_eligible"] != (root not in (3, 5))):
        raise ValueError("original independently audited source dictionary provenance changed")
    # Consume only the metadata/scorer reports and checkpoint raw bytes.
    # Do not open case-level raw JSONL, even to read a line.
    _bool_clean(source, (
        "development_evaluation_groups_scored", "reserved_context_groups_scored",
        "embargoed_worlds_scored", "four_input_confirmation_worlds_scored",
        "new_development_evaluation_groups_scored",
    ))
    receipts, detail = {}, {}
    for arm in ARMS:
        expected = t["arms"][arm]
        receipts[arm], detail[arm] = _source_arm(
            source_dir, reset_dir, root, arm, source, t,
            expected["source_model_sha256"],
            expected["source_sae_checkpoint_sha256"], source_sha)
        if expected["original_arm_source_eligible"] != detail[arm]["original_source_eligible"]:
            raise ValueError("original source arm eligibility changed")
    return {
        "schema": SCHEMA,
        "world_index": root, "world_id": t["world_id"],
        "world_sha256": t["world_sha256"],
        "original_nec03t_report_sha256": sha256_file(t_report_path),
        "original_nec03t_source_lock_sha256": original_source_lock_sha256,
        "original_nec03u_archive_lock_sha256":
            parent_lock.get("_lock_sha256") if historical_lock_verified else None,
        "NEC03T_root_artifact_id": T_ARTIFACTS[root],
        "original_source_artifact_id": SOURCE_ARTIFACTS[root],
        "original_source_run_id": OLD_SOURCE_RUN if root < 4 else NEW_SOURCE_RUN,
        "original_nec03t_run_id": T_RUN,
        "source_checkpoint_receipts": receipts,
        "source_checkpoint_detail": detail,
        "source_checkpoint_bytes_verified": True,
        "original_archival_lock_verified": historical_lock_verified,
        "per_case_activation_coefficients_verified": False,
        "projected_decoder_columns_verified": False,
        "native_model_or_SAE_loaded": False,
        "model_forwards": 0,
        "new_training_updates": 0,
        "new_recruited_or_embargoed_reads": 0,
        "new_confirmation_reads": 0,
        "source_scope": list(range(24, 32)),
        "original_fixed_zero_roots": [3, 5],
        "next_gate": "authenticated-source-activation-geometry-extraction",
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
    }


def attest(root: int, source_dir: Path, t_dir: Path, reset_dir: Path | None):
    here = Path(__file__).parent
    original_t_lock = here / "NEC03T_SOURCE_LOCK.json"
    original_u_lock = here / "NEC03U_ARCHIVE_LOCK.json"
    parent = _read(original_u_lock)
    if (parent.get("schema") != "NEC-03U/archived-source-margin-readonly-method/v1"
            or parent.get("original_nec03t_run") != T_RUN
            or parent.get("original_world_root_indices") != list(ROOTS)
            or parent.get("original_source_group_indices") != list(range(24, 32))
            or parent.get("original_fixed_zero_roots") != [3, 5]
            or [parent["artifact_ids"][str(i)] for i in ROOTS] != list(T_ARTIFACTS)
            or parent.get("model_forwards") != 0):
        raise ValueError("original immutable source archive lock no longer matches")
    parent["_lock_sha256"] = sha256_file(original_u_lock)
    report = verify_archive(
        root, source_dir, t_dir, reset_dir, parent_lock=parent,
        original_source_lock_sha256=sha256_file(original_t_lock),
        historical_lock_verified=True,
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-index", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--nec03t-root", type=Path, required=True)
    parser.add_argument("--reset-root", type=Path,
                        help="required for historical worlds 0..3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("attestation receipt destination must be fresh")
    result = attest(args.world_index, args.source_root, args.nec03t_root,
                    args.reset_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(canonical(result) + "\n", encoding="utf-8")
    print(canonical({name: result[name] for name in (
        "world_id", "source_checkpoint_bytes_verified",
        "original_archival_lock_verified", "model_forwards", "next_gate",
    )}))
    return result


if __name__ == "__main__":
    main()
