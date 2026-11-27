"""Append-only, hash-chained JSONL experiment ledger. Streams; never loads all records."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

GENESIS = "0" * 64


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


class Ledger:
    """Each line: {"seq", "kind", "prev", "body", "hash"}; hash = sha256(seq, kind, prev, body)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.seq, self.head = 0, GENESIS
        for rec in self.iter_verified():
            self.seq, self.head = rec["seq"] + 1, rec["hash"]

    def append(self, kind: str, body: dict) -> dict:
        rec = {"seq": self.seq, "kind": kind, "prev": self.head, "body": body}
        rec["hash"] = sha256(rec)
        line = json.dumps(rec, sort_keys=True, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self.seq, self.head = self.seq + 1, rec["hash"]
        return rec

    def iter_verified(self) -> Iterator[dict]:
        prev, seq = GENESIS, 0
        with self.path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    # Torn final write from an emergency stop: tolerate only at EOF.
                    if f.readline():
                        raise ValueError(f"corrupt ledger line {lineno}")
                    return
                body = {k: rec[k] for k in ("seq", "kind", "prev", "body")}
                if rec["seq"] != seq or rec["prev"] != prev or sha256(body) != rec["hash"]:
                    raise ValueError(f"ledger chain broken at line {lineno}")
                yield rec
                prev, seq = rec["hash"], seq + 1

    def kinds(self, kind: str) -> Iterator[dict]:
        for rec in self.iter_verified():
            if rec["kind"] == kind:
                yield rec["body"]

    def last(self, kind: str) -> dict | None:
        out = None
        for body in self.kinds(kind):
            out = body
        return out


def append_jsonl(path: Path, record: dict) -> None:
    with Path(path).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                yield json.loads(raw)


def file_sha256(path: Path) -> str:
    """LF-normalised digest so receipts hash identically across Windows and CI checkouts."""
    h, carry = hashlib.sha256(), b""
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            block = carry + block
            carry = b"\r" if block.endswith(b"\r") else b""
            if carry:
                block = block[:-1]
            h.update(block.replace(b"\r\n", b"\n"))
    h.update(carry)
    return h.hexdigest()
