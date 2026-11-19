"""Lossless numeric capture with content-addressed NPY payloads.

Logical names remain independent measurements. Equal dtype, shape and value
bytes may share a payload; their receipt call counts stay separate.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np


SCHEMA = "LGA10D/aliased-npz-v1"
_LOGICAL = re.compile(r"[A-Za-z0-9_]+\Z")
_PHYSICAL = re.compile(r"a_[0-9a-f]{64}\Z")


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _alias_sha(aliases):
    canonical = json.dumps(aliases, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _encode(value):
    array = np.array(value, copy=True, order="C", subok=False)
    if array.dtype.hasobject or array.dtype.kind not in "biufc":
        raise ValueError("capture requires numeric arrays without objects")
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return array, buffer.getvalue()


def write_capture(path, logical_arrays):
    """Write each exact numeric payload once and return its complete alias map."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("a first capture cannot be overwritten")
    aliases, payloads, payload_bytes = {}, {}, {}
    logical_bytes = logical_npy_bytes = 0
    for name in sorted(logical_arrays):
        if not isinstance(name, str) or not _LOGICAL.fullmatch(name):
            raise ValueError("invalid logical capture key")
        array, payload = _encode(logical_arrays[name])
        key = "a_" + hashlib.sha256(payload).hexdigest()
        if key in payloads and payloads[key] != payload:
            raise ArithmeticError("content-address collision")
        payloads[key] = payload
        payload_bytes[key] = array.nbytes
        aliases[name] = key
        logical_bytes += array.nbytes
        logical_npy_bytes += len(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key in sorted(payloads):
            archive.writestr(key + ".npy", payloads[key])
    unique_bytes = sum(payload_bytes.values())
    unique_npy_bytes = sum(map(len, payloads.values()))
    return {
        "schema": SCHEMA, "raw_sha256": _file_sha(path),
        "logical_to_physical": aliases, "alias_map_sha256": _alias_sha(aliases),
        "logical_array_count": len(aliases), "unique_array_count": len(payloads),
        "logical_payload_bytes": logical_bytes, "unique_payload_bytes": unique_bytes,
        "duplicate_payload_bytes_saved": logical_bytes - unique_bytes,
        "logical_npy_bytes": logical_npy_bytes, "unique_npy_bytes": unique_npy_bytes,
        "raw_file_bytes": path.stat().st_size,
    }


class LogicalCapture:
    """Verified logical array view; hash the stored NPY bytes across versions."""

    def __init__(self, path, metadata):
        path = Path(path)
        if metadata.get("schema") != SCHEMA:
            raise ValueError("unknown capture schema")
        if _file_sha(path) != metadata.get("raw_sha256"):
            raise ValueError("raw capture digest differs")
        aliases = metadata.get("logical_to_physical")
        if not isinstance(aliases, dict) or any(
                not isinstance(name, str) or not _LOGICAL.fullmatch(name)
                or not isinstance(key, str) or not _PHYSICAL.fullmatch(key)
                for name, key in aliases.items()):
            raise ValueError("invalid alias map")
        if metadata.get("alias_map_sha256") != _alias_sha(aliases):
            raise ValueError("alias map digest differs")
        unique = set(aliases.values())
        if (metadata.get("logical_array_count") != len(aliases)
                or metadata.get("unique_array_count") != len(unique)):
            raise ValueError("capture counts differ")
        self._aliases = dict(aliases)
        self._arrays = {}
        lengths = {}
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != {key + ".npy" for key in unique}:
                raise ValueError("physical payload names differ")
            for key in sorted(unique):
                payload = archive.read(key + ".npy")
                if "a_" + hashlib.sha256(payload).hexdigest() != key:
                    raise ValueError("physical payload content digest differs")
                array = np.load(io.BytesIO(payload), allow_pickle=False)
                if (not isinstance(array, np.ndarray) or array.dtype.hasobject
                        or array.dtype.kind not in "biufc"):
                    raise ValueError("physical payload is not a numeric array")
                array.setflags(write=False)
                self._arrays[key] = array
                lengths[key] = len(payload)
        expected = {
            "logical_payload_bytes": sum(self._arrays[key].nbytes for key in aliases.values()),
            "unique_payload_bytes": sum(array.nbytes for array in self._arrays.values()),
            "logical_npy_bytes": sum(lengths[key] for key in aliases.values()),
            "unique_npy_bytes": sum(lengths.values()), "raw_file_bytes": path.stat().st_size,
        }
        expected["duplicate_payload_bytes_saved"] = expected["logical_payload_bytes"] - expected["unique_payload_bytes"]
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("capture byte counts differ")
        self.files = list(aliases)

    def __getitem__(self, key):
        return self._arrays[self._aliases[key]]

    def __contains__(self, key):
        return key in self._aliases

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._arrays.clear()


def open_capture(path, metadata):
    return LogicalCapture(path, metadata)
