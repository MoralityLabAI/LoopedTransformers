"""Atomic checkpoints paired with a hash-chained, append-only episode ledger."""
from __future__ import annotations

import io
import json
import sqlite3
from contextlib import contextmanager

import torch

from .data import canonical, digest


def pack(state):
    stream = io.BytesIO()
    torch.save(state, stream)
    return stream.getvalue()


def unpack(payload):
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


@contextmanager
def open_store(path, *, write=False):
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=0)
    else:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        if write:
            conn.execute("PRAGMA synchronous=FULL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS manifest (id INTEGER PRIMARY KEY CHECK(id=0), json TEXT);
                CREATE TABLE IF NOT EXISTS ledger (seq INTEGER PRIMARY KEY, hash TEXT UNIQUE, json TEXT);
                CREATE TABLE IF NOT EXISTS weights (hash TEXT PRIMARY KEY, state BLOB);
                CREATE TABLE IF NOT EXISTS checkpoint (id INTEGER PRIMARY KEY CHECK(id=0), state BLOB);
            """)
        yield conn
    finally:
        conn.close()


def initialize(conn, manifest):
    raw = canonical(manifest)
    with conn:
        conn.execute("INSERT OR IGNORE INTO manifest VALUES (0, ?)", (raw,))
    if read_manifest(conn) != json.loads(raw):
        raise ValueError("resume parameters, frozen curricula, source or executable identity changed")


def read_manifest(conn):
    result = conn.execute("SELECT json FROM manifest WHERE id=0").fetchone()
    if result is None:
        raise ValueError("missing session manifest")
    return json.loads(result[0])


def append(conn, row, state=None, backbone=None):
    previous = conn.execute("SELECT seq, hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
    seq, parent = (previous[0] + 1, previous[1]) if previous else (0, None)
    body = {**row, "seq": seq, "parent_sha256": parent}
    hash_ = digest(body)
    conn.execute("INSERT INTO ledger VALUES (?, ?, ?)", (seq, hash_, canonical(body)))
    if state is not None:
        conn.execute("INSERT OR REPLACE INTO checkpoint VALUES (0, ?)", (pack(state),))
    if backbone is not None:
        conn.execute("INSERT OR IGNORE INTO weights VALUES (?, ?)", (backbone[0], pack(backbone[1])))


def ledger(conn):
    parent = None
    for expected, (seq, hash_, raw) in enumerate(conn.execute("SELECT * FROM ledger ORDER BY seq")):
        item = json.loads(raw)
        if (seq != expected or item["seq"] != seq or item["parent_sha256"] != parent
                or digest(item) != hash_):
            raise ValueError("episode ledger hash chain changed")
        parent = hash_
        yield item


def checkpoint(conn):
    row = conn.execute("SELECT state FROM checkpoint WHERE id=0").fetchone()
    return unpack(row[0]) if row is not None else None


def load_backbone(conn, name):
    row = conn.execute("SELECT state FROM weights WHERE hash=?", (name,)).fetchone()
    if row is None:
        raise ValueError("checkpoint references a missing backbone")
    return unpack(row[0])
