"""Verify and assemble the four published ROC-02A frozen NRF byte archives.

Pure Python stdlib; requires no PyTorch. Uses the original per-cell NRF-11
receipts to check the binary SHA-256 values of all twelve .pt checkpoints and
twelve .pt learned observer adapters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_and_unpack(archive_dir: Path, destination: Path, index_path: Path) -> dict:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index["study_id"] != "ROC-02A-RETRO-frozen-NRF-rank-causal-pilot-v1":
        raise ValueError("unexpected binary manifest study")
    archives = index["archives"]
    if len(archives) != 4 or len({x["name"] for x in archives}) != 4:
        raise ValueError("expected four distinct registered archive parts")
    destination.mkdir(parents=True, exist_ok=True)
    extracted: set[str] = set()
    for item in archives:
        path = archive_dir / item["name"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"archive digest mismatch: {item['name']}")
        with ZipFile(path) as z:
            if z.testzip() is not None:
                raise RuntimeError(f"corrupt ZIP archive: {item['name']}")
            members = z.infolist()
            if len(members) != item["members"]:
                raise RuntimeError(f"archive member count mismatch: {item['name']}")
            for member in members:
                label = member.filename
                pure = PurePosixPath(label)
                if (member.is_dir() or pure.is_absolute() or ".." in pure.parts
                        or not pure.parts or label in extracted
                        or "\\" in label):
                    raise RuntimeError(f"unsafe or duplicate archive member: {label}")
                target = destination.joinpath(*pure.parts)
                if target.exists():
                    raise FileExistsError(f"destination already has {label}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))
                extracted.add(label)
    seeds = [131, 137, 139, 149, 151, 157]
    if len(extracted) != 36:
        raise RuntimeError(f"expected 36 distinct source files, got {len(extracted)}")
    verified = []
    for depth in (16, 32):
        for seed in seeds:
            receipt = destination / f"nrf11_work/receipts/nrf11_R{depth}_S{seed}.json"
            ck = destination / f"nrf09_local/checkpoints/nrf09_tied_R{depth}_S{seed}_steps32.pt"
            adapter = destination / f"nrf11_work/receipts/observers/nrf11_observer_R{depth}_S{seed}.pt"
            record = json.loads(receipt.read_text(encoding="utf-8"))
            ck_hash, ad_hash = sha256(ck), sha256(adapter)
            if ck_hash != record["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint digest mismatch R{depth} S{seed}")
            if ad_hash != record["observer_adapter_sha256"]:
                raise RuntimeError(f"observer digest mismatch R{depth} S{seed}")
            if record["seed"] != seed or record["depth"] != depth:
                raise RuntimeError(f"incorrect receipt identity R{depth} S{seed}")
            verified.append({"depth": depth, "seed": seed,
                             "checkpoint_sha256": ck_hash,
                             "observer_adapter_sha256": ad_hash})
    return {"study_id": index["study_id"], "archives": 4,
            "verified_checkpoints": len(verified),
            "verified_adapters": len(verified),
            "verified_receipts": len(verified),
            "replay_directory": str(destination), "records": verified}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-dir", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(verify_and_unpack(args.archive_dir, args.destination, args.index),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
