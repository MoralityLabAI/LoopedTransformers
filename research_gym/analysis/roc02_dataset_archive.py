"""Deterministic ROC-02 source bundle packer and SHA-verified public loader."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
from io import BytesIO
import gzip
import hashlib
import json
import tarfile

NAMES = ("train_inputs.jsonl", "calibration_inputs.jsonl",
         "sealed_test_inputs.jsonl", "invalid_oracle_fixtures.jsonl")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pack(source: Path, dest: Path) -> dict:
    manifest = json.loads((source / "source_manifest.json").read_text())
    for name in NAMES:
        if not (source / name).is_file():
            raise ValueError("missing declared split: " + name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as h:
        with gzip.GzipFile(fileobj=h, mode="wb", filename="", mtime=0) as g:
            with tarfile.open(fileobj=g, mode="w") as tar:
                for name in NAMES:
                    raw = (source / name).read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(raw)
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    tar.addfile(info, BytesIO(raw))
    verify = verify_archive(dest, manifest)
    verify["archive_sha256"] = sha(dest.read_bytes())
    return verify


def verify_archive(path: Path, manifest: dict) -> dict:
    with tarfile.open(path, mode="r:gz") as tar:
        members = tar.getmembers()
        if len(members) != len(NAMES) or {m.name for m in members} != set(NAMES):
            raise ValueError("unexpected registered archive members")
        hashes = {}
        for m in members:
            parts = PurePosixPath(m.name)
            if (not m.isfile() or parts.is_absolute() or ".." in parts.parts or
                    len(parts.parts) != 1):
                raise ValueError("unsafe archive member")
            raw = tar.extractfile(m).read()
            hashes[m.name] = sha(raw)
            if m.name == "sealed_test_inputs.jsonl":
                if any("label" in json.loads(line) for line in raw.splitlines()):
                    raise AssertionError("sealed source exposes labels")
    for cfg in manifest["splits"].values():
        if hashes[cfg["file_name"]] != cfg["file_sha256"]:
            raise RuntimeError("source split digest mismatch")
    if hashes["invalid_oracle_fixtures.jsonl"] != manifest["invalid"]["sha256"]:
        raise RuntimeError("invalid fixture digest mismatch")
    return {"source_study": manifest["study"], "member_sha256": hashes,
            "files": len(hashes), "valid_cases": sum(x["n"] for x in manifest["splits"].values()),
            "invalid_fixtures": manifest["invalid"]["n"]}
