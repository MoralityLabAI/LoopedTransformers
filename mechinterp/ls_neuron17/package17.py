"""Deterministically package existing LS17 receipts; no model computation."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import zipfile


ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    locked = {}
    for line in (ROOT / "lock.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        assert digest((ROOT / relative).read_bytes()) == expected, relative
        locked[relative] = expected
    parent = json.loads((ROOT / "parent_manifest.json").read_text())
    for relative, expected in parent["files"].items():
        if isinstance(expected, dict):
            expected = expected["sha256"]
        assert digest((ROOT / relative).read_bytes()) == expected, relative
    development = json.loads((ROOT / "development_manifest.json").read_text())
    for relative, expected in development["files"].items():
        assert digest((ROOT / relative).read_bytes()) == expected["sha256"], relative

    names = set()
    for folder in ("initial", "common", "beam", "results", "development", "provenance"):
        for file in (ROOT / folder).rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                names.add(file.relative_to(ROOT).as_posix())
    names.update((
        "baseline_certificates.json", "parent_manifest.json", "development_manifest.json",
        "development_validation.json", "run_manifest.json", "execution_receipt.json",
        "runtime17.json", "freeze_receipt.json", "report_layout_revision.json",
        "search.log", "audit.log", "statistics.log", "report.log", "report_layout_revision.log",
        "search_exit.status", "audit_exit.status", "statistics_exit.status", "report_exit.status",
    ))
    payload = {name: (ROOT / name).read_bytes() for name in sorted(names)}
    manifest = {
        "experiment": "LS-NEURON-17", "scope": "EXPOSED_DEVELOPMENT_ONLY",
        "source_lock_sha256": digest((ROOT / "lock.sha256").read_bytes()),
        "files": {name: {"bytes": len(data), "sha256": digest(data)} for name, data in payload.items()},
        "format": "ZIP_DEFLATED level 9; fixed timestamp 2026-11-25T00:00:00; Unix mode 0644",
        "manifest_self_hash_excluded": True,
        "archive_scope": "Raw initial, shared bootstrap, archived beam and new multistart receipts; old-seed development data; provenance, runtime and execution logs",
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (ROOT / "receipt_manifest.json").write_bytes(encoded)
    payload["receipt_manifest.json"] = encoded
    with zipfile.ZipFile(ROOT / "receipts.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(payload.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 11, 25, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    raw_certificates = (ROOT / "certificates.json").read_bytes()
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0, compresslevel=9) as stream:
        stream.write(raw_certificates)
    (ROOT / "certificates.json.gz").write_bytes(compressed.getvalue())
    assert gzip.decompress(compressed.getvalue()) == raw_certificates
    with zipfile.ZipFile(ROOT / "receipts.zip") as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(payload)
        for name, data in payload.items():
            assert z.read(name) == data, name

    inventory = {
        "experiment": "LS-NEURON-17", "source_files_verified": len(locked),
        "parent_inputs_verified": len(parent["files"]),
        "development_files_verified": len(development["files"]),
        "receipts_archive_entries": len(payload),
        "receipts_uncompressed_bytes": sum(map(len, payload.values())),
        "certificates_uncompressed_bytes": len(raw_certificates),
        "certificates_uncompressed_sha256": digest(raw_certificates),
        "archive_roundtrip_verified": True,
        "artifacts": {name: {"bytes": (ROOT / name).stat().st_size,
                             "sha256": digest((ROOT / name).read_bytes())}
                      for name in ("receipts.zip", "certificates.json.gz", "receipt_manifest.json",
                                   "README.md", "audit_results.json", "report_data.json",
                                   "figure17.png", "figure17.svg")},
    }
    (ROOT / "package_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
