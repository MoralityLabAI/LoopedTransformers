"""Fetch an immutable recovery archive, verify it, and inventory exact bytes.

Only Python's standard library is needed. Extraction never executes recovered
code or unpickles checkpoints. Use weights_only=True in separate model tooling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, spec: dict) -> None:
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Archive size mismatch: {path.stat().st_size}")
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise ValueError(f"Archive SHA-256 mismatch: {actual}")


def fetch_archive(path: Path, spec: dict) -> None:
    if path.exists():
        verify_archive(path, spec)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rename only after the complete download verifies. No partial archive can
    # accidentally become an input to the analysis.
    part = path.with_suffix(path.suffix + ".partial")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(spec["url"], timeout=120) as response:
                with part.open("wb") as output:
                    shutil.copyfileobj(response, output, 8 * 1024 * 1024)
            verify_archive(part, spec)
            part.replace(path)
            return
        except (OSError, ValueError):
            part.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def validated_member(info: zipfile.ZipInfo) -> PurePosixPath:
    path = PurePosixPath(info.filename)
    if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
            or not path.parts or ":" in path.parts[0]):
        raise ValueError(f"Unsafe archive path: {info.filename!r}")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError(f"Symlink member rejected: {info.filename!r}")
    return path


def inventory_archive(archive: Path, spec: dict, destination: Path | None) -> dict:
    verify_archive(archive, spec)
    members = []
    names = set()
    if destination is not None:
        destination = destination.resolve()
        destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as recovery:
        infos = recovery.infolist()
        if len(infos) != spec["members"]:
            raise ValueError("Archive member count differs from frozen specification")
        if sum(info.file_size for info in infos) != spec["uncompressed_bytes"]:
            raise ValueError("Archive expanded size differs from frozen specification")
        for info in infos:
            relative = validated_member(info)
            if info.filename in names:
                raise ValueError(f"Duplicate archive path: {info.filename}")
            names.add(info.filename)
            if info.is_dir():
                continue
            digest = hashlib.sha256()
            total = 0
            output = None
            temporary = None
            target = destination / relative if destination is not None else None
            if target is not None:
                if not target.resolve().is_relative_to(destination):
                    raise ValueError("Extraction target escapes destination")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    fd, temporary = tempfile.mkstemp(prefix=".recovery-", dir=target.parent)
                    output = os.fdopen(fd, "wb")
            try:
                with recovery.open(info) as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                        total += len(chunk)
                        if output is not None:
                            output.write(chunk)
                if output is not None:
                    output.close()
                    output = None
                if total != info.file_size:
                    raise ValueError(f"Expanded size mismatch: {info.filename}")
                hexdigest = digest.hexdigest()
                if target is not None:
                    if temporary is not None:
                        Path(temporary).replace(target)
                        temporary = None
                        target.chmod(0o444)
                    elif sha256_file(target) != hexdigest:
                        raise ValueError(f"Existing recovered file was modified: {info.filename}")
                members.append({
                    "path": info.filename,
                    "bytes": total,
                    "compressed_bytes": info.compress_size,
                    "sha256": hexdigest,
                    "crc32": f"{info.CRC:08x}",
                })
            finally:
                if output is not None:
                    output.close()
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
        embedded = list(csv.DictReader(io.StringIO(recovery.read("MANIFEST.csv").decode("utf-8-sig")))) if "MANIFEST.csv" in names else []
    indexed = {member["path"]: member for member in members}
    for record in embedded:
        member = indexed.get(record["archive_path"])
        if member is None or member["sha256"] != record["sha256"] or member["bytes"] != int(record["size_bytes"]):
            raise ValueError(f"Embedded manifest mismatch: {record['archive_path']}")
    return {
        "schema_version": 1,
        "archive": spec,
        "verified": True,
        "member_count": len(members),
        "embedded_manifest_entries_verified": len(embedded),
        "total_uncompressed_bytes": sum(member["bytes"] for member in members),
        "members": sorted(members, key=lambda member: member["path"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "archive.json")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if not args.offline:
        fetch_archive(args.archive, spec)
    manifest = inventory_archive(args.archive, spec, args.extract)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sha256": spec["sha256"], "member_count": manifest["member_count"],
                      "total_uncompressed_bytes": manifest["total_uncompressed_bytes"]}))


if __name__ == "__main__":
    main()
