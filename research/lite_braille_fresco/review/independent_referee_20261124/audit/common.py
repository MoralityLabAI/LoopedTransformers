"""Shared helpers for the independent LBF-14..17 referee audit.

Custody rules enforced here:
  * Nothing under ``results/`` is ever written to. Archives are decompressed into a
    scratch directory supplied by the caller (default: a fresh temp directory).
  * The sealed ``ordinary`` / ``ood`` cohorts are never parsed. ``sealed_stream_sha256``
    hashes the decompressed byte stream without retaining or decoding it.
  * No checkpoint is modified and no optimiser is ever constructed.

All paths are passed in explicitly; nothing here depends on the current directory.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

REVIEW_COMMIT = "3d8243818858c19c5031c36e2bf7e328f831cf25"
PINNED_SIM_COMMIT = "cd9ac784c5804ec2fe595bb6a01d2697fc016610"
PINNED_SIM_BLOB = "204a1d012a4ebe8e80d8bd722590742b7e7f02ba"
SEALED_SPLITS = ("ordinary", "ood")
OPEN_SPLITS = ("train", "calibration")


def lbf_dir(repo: Path) -> Path:
    return Path(repo) / "research" / "lite_braille_fresco"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gz_stream_sha256(path: Path) -> str:
    """SHA-256 of the decompressed stream. Bytes are hashed and discarded, never decoded."""
    h = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(("blob " + str(len(raw)) + "\0").encode() + raw).hexdigest()


def read_jsonl_gz(path: Path) -> list[dict]:
    name = Path(path).name
    for sealed in SEALED_SPLITS:
        if name.startswith(sealed + "."):
            raise PermissionError(f"refusing to parse sealed cohort file {name}")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def scratch_dir(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path(tempfile.mkdtemp(prefix="lbf_audit_"))


def load_modules(repo: Path, pinned_sim: Path):
    """Import the registered LBF-17 module and the pinned rewrite simulator (hash-checked)."""
    d = lbf_dir(repo)
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    import lbf15_meeting_fresco as m15  # noqa: E402
    import lbf16_equivariant_fresco as m16  # noqa: E402
    import lbf17_relational_fresco as m17  # noqa: E402

    raw = Path(pinned_sim).read_bytes()
    if git_blob_sha1(raw) != PINNED_SIM_BLOB:
        raise ValueError("pinned rewrite simulator blob hash mismatch")
    rg = m17.base.load_upstream(Path(pinned_sim))
    return rg, m15, m16, m17


def load_open_split(repo: Path, rg, split: str, generator_dir: str = "lbf17"):
    """Parse an OPEN split (train / calibration) after verifying its registered hash."""
    if split not in OPEN_SPLITS:
        raise PermissionError("only train/calibration may be parsed by the audit")
    suite = lbf_dir(repo) / "results" / generator_dir / "suite"
    manifest = json.loads((suite / "manifest.json").read_text(encoding="utf-8"))
    gz = suite / f"{split}.jsonl.gz"
    got = gz_stream_sha256(gz)
    want = manifest["artifacts"][f"{split}.jsonl"]
    if got != want:
        raise ValueError(f"{split}.jsonl hash mismatch: {got} != {want}")
    rows = read_jsonl_gz(gz)
    cases = [rg.Case.parse(r) for r in rows]
    if len(cases) != manifest["counts"][split]:
        raise ValueError("unexpected split size")
    return cases, manifest


def load_checkpoint(repo: Path, m17, seed: int, arch: str, scratch: Path, generator_dir: str = "lbf17"):
    """Decompress an archived checkpoint into scratch, verify its registered SHA-256, load frozen."""
    import torch

    mdir = lbf_dir(repo) / "results" / generator_dir / f"{seed}_{arch}"
    rec = json.loads((mdir / "manifest.json").read_text(encoding="utf-8"))
    dst = Path(scratch) / f"{seed}_{arch}.checkpoint.pt"
    if not dst.exists():
        with gzip.open(mdir / "checkpoint.pt.gz", "rb") as src:
            dst.write_bytes(src.read())
    got = sha256_file(dst)
    if got != rec["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch for {seed}:{arch}")
    model, _ = m17.new_lbf17_model(seed, arch)
    ck = torch.load(dst, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["weights"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # NOTE: the registered "final" fingerprint is sha256(torch.save(state_dict)), which depends on
    # the torch serialisation version and is therefore not portable.  The checkpoint FILE hash is
    # the authoritative custody check here; we additionally record a canonical tensor hash.
    rec = dict(rec)
    rec["registered_fingerprint_reproduced"] = (m17.base._model_fingerprint(model) == rec["final"])
    rec["canonical_tensor_sha256"] = canonical_state_hash(model)
    rec["loaded_parameters"] = sum(p.numel() for p in model.parameters())
    if rec["loaded_parameters"] != rec["parameters"]:
        raise ValueError(f"parameter count mismatch for {seed}:{arch}")
    return model, rec


def canonical_state_hash(model) -> str:
    """Serialisation-independent hash: sorted (name, dtype, shape, little-endian bytes)."""
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        t = t.detach().cpu().contiguous()
        h.update(name.encode() + b"\0" + str(t.dtype).encode() + b"\0" + str(tuple(t.shape)).encode() + b"\0")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def bfs_ball(rules, w0: str, k_max: int) -> dict[str, int]:
    """Exact concrete <=k-step reachability under symmetric length-preserving rules."""
    ors = [(u, v) for u, v in rules if u != v] + [(v, u) for u, v in rules if u != v]
    seen = {w0: 0}
    frontier = [w0]
    for d in range(1, k_max + 1):
        nxt = []
        for w in frontier:
            for lhs, rhs in ors:
                ll = len(lhs)
                start = w.find(lhs)
                while start != -1:
                    z = w[:start] + rhs + w[start + ll:]
                    if z not in seen:
                        seen[z] = d
                        nxt.append(z)
                    start = w.find(lhs, start + 1)
        frontier = nxt
        if not frontier:
            break
    return seen


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_single_thread():
    import torch

    torch.set_num_threads(1)
    try:
        torch.backends.mkldnn.enabled = False
    except Exception:  # pragma: no cover
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
