"""Verified routing inputs and disk-backed, pass-conditioned causal windows."""
import json
import shutil
import tarfile
from dataclasses import dataclass
from itertools import groupby
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np

from gym_manifest import entries, sha, verify
from routing import parse_visit

ARMS = ("generated", "reference_forced")
HORIZONS = (0, 32, 64, 128)
LAYERS, PASSES, EXPERTS = 24, 3, 128
ACTIVE = np.ones((LAYERS, PASSES), dtype=bool)
ACTIVE[:6, 1:] = ACTIVE[18:, 1:] = False
PASS_COUNT = ACTIVE.sum(axis=1)
PILOT_BYTES = 33_104_911
PILOT_SHA256 = "16d1930ec23f91e28003bb654011215438116fa15c5b609a4566abf39a02b5b4"
SYNTHETIC = "synthetic_schema_fixture"


def write_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def extract_pilot(archive, destination):
    if archive.stat().st_size != PILOT_BYTES or sha(archive) != PILOT_SHA256:
        raise ValueError("pilot archive size or SHA-256 differs from the published receipt")
    destination.mkdir(parents=True)
    try:
        seen = set()
        with tarfile.open(archive, "r:gz") as bundle:
            top = None
            for member in bundle:
                parts = PurePosixPath(member.name).parts
                if (not parts or member.name.startswith("/") or
                        any(p in ("..", ".") or ":" in p or "\\" in p or
                            PureWindowsPath(p).is_reserved() for p in parts)):
                    raise ValueError("unsafe archive path")
                if top is None:
                    top = parts[0]
                if parts[0] != top or (not member.isdir() and not member.isfile()):
                    raise ValueError("expected one archive directory with regular files")
                path = destination.joinpath(*parts[1:])
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                if len(parts) < 2 or path in seen:
                    raise ValueError("invalid or duplicate archive file")
                seen.add(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1 << 20)
        if len(verify(destination)["files"]) != 18:
            raise ValueError("pilot manifest must list exactly 18 files")
        receipt = {
            "archive_bytes": PILOT_BYTES, "archive_sha256": PILOT_SHA256,
            "manifest_sha256": sha(destination / "manifest.json"), "verified_files": 18,
        }
    except BaseException:
        shutil.rmtree(destination)
        raise
    return receipt


def source_receipt(root, synthetic):
    if synthetic:
        manifest = read_json(root / "manifest.json")
        if manifest.get("kind") != SYNTHETIC or manifest.get("files") != entries(root):
            raise ValueError(f"synthetic fixture integrity mismatch: {root}")
    else:
        manifest = verify(root)
        online = read_json(root / "online_receipt.json")
        if online["status"] != "complete" or any(not online["arms"][arm]["complete"] for arm in ARMS):
            raise ValueError(f"incomplete teacher capture: {root}")
    return {"manifest_sha256": sha(root / "manifest.json"), "files": len(manifest["files"])}


def json_rows(path):
    with path.open(encoding="utf8") as rows:
        for row in rows:
            if row.strip():
                yield json.loads(row)


@dataclass(frozen=True)
class Turn:
    root: str
    turn: str
    family: str
    split: str
    prompt: int
    tokens: int

    def bounds(self, horizon):
        width = horizon or self.tokens - self.prompt
        for start in range(self.prompt, self.tokens, width):
            yield start, min(self.tokens, start + width)


@dataclass(frozen=True)
class Window:
    turn: Turn
    start: int
    end: int


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray
    windows: list

    def close(self):
        # Each array owns just one mapping. Explicit close also permits Windows cleanup.
        self.x._mmap.close()
        self.y._mmap.close()


def read_rosters(sources, arm):
    rosters = []
    seen_roots, seen_families = set(), set()
    root_splits, family_splits = {}, {}
    for source in sources:
        roster = {}
        for row in json_rows(source / arm / "roster.jsonl"):
            turn = Turn(row["root"], row["turn"], row["family"], row["split"],
                        row["prompt_len"], len(row["token_ids"]))
            ids = turn.root, turn.turn
            if (not all(isinstance(s, str) and s for s in (*ids, turn.family))
                    or turn.split not in ("train", "dev") or type(turn.prompt) is not int
                    or not 0 < turn.prompt < turn.tokens or ids in roster
                    or turn.root in seen_roots or turn.family in seen_families):
                raise ValueError(f"invalid or duplicate roster entry: {ids}")
            for value, splits in ((turn.root, root_splits), (turn.family, family_splits)):
                if splits.setdefault(value, turn.split) != turn.split:
                    raise ValueError(f"split leakage: {value}")
            roster[ids] = turn
        if not roster:
            raise ValueError(f"empty roster: {source / arm}")
        seen_roots.update(turn.root for turn in roster.values())
        seen_families.update(turn.family for turn in roster.values())
        rosters.append(roster)
    if set(root_splits.values()) != {"train", "dev"}:
        raise ValueError("need both train and dev roots")
    return rosters


def trace_turns(path, roster, guard):
    seen = set()
    for key, rows in groupby(json_rows(path), lambda row: (row["root"], row["turn"])):
        if key not in roster or key in seen:
            raise ValueError(f"unregistered or noncontiguous trace turn: {key}")
        turn = roster[key]
        if turn.tokens * LAYERS * PASSES * EXPERTS * 4 > 256 * (1 << 20):
            raise ValueError(f"turn exceeds the bounded trace buffer: {key}")
        mass = np.zeros((turn.tokens, LAYERS, PASSES, EXPERTS), dtype=np.float32)
        visits = np.zeros(mass.shape[:-1], dtype=bool)
        frequency = np.zeros((LAYERS, EXPERTS), dtype=np.int64)
        for i, row in enumerate(rows):
            (_, _, token, layer, pass_id), ids, weights = parse_visit(row)
            if (token >= turn.tokens or visits[token, layer, pass_id] or
                    row.get("phase") != ("prompt" if token < turn.prompt else "continuation")):
                raise ValueError(f"duplicate or out-of-phase trace visit: {key}")
            mass[token, layer, pass_id, ids] = weights
            visits[token, layer, pass_id] = True
            if turn.split == "train" and token >= turn.prompt:
                frequency[layer, ids] += 1
            if i % 2048 == 0:
                guard.check()
        if not np.all(visits == ACTIVE):
            raise ValueError(f"missing native visits: {key}")
        seen.add(key)
        yield turn, mass, frequency
    if seen != roster.keys():
        raise ValueError(f"roster/trace mismatch: {path}")


def normalize(mass):
    return mass / np.maximum(mass.sum(axis=-1, keepdims=True), 1e-12)


def collapse(passes):
    """Mean over the actual passes of each physical layer."""
    return passes.sum(axis=-2) / PASS_COUNT.astype(np.float32)[:, None]


def write_windows(turn, mass, horizon, start_index, dataset):
    prompt = mass[:turn.prompt].sum(axis=0)
    history = prompt.copy()
    prompt = normalize(prompt)
    for i, (start, end) in enumerate(turn.bounds(horizon), start_index):
        dataset.x[i, ..., :EXPERTS] = prompt
        dataset.x[i, ..., EXPERTS:] = normalize(history)
        target = mass[start:end].sum(axis=0)
        dataset.y[i] = normalize(target)
        history += target


def verify_original(path, horizon, dataset, window_index, guard):
    """Cross-check the reconstruction against the supplied pass-collapsed arrays."""
    with np.load(path, allow_pickle=False) as original:
        if int(original["horizon"]) != horizon:
            raise ValueError(f"wrong demand horizon: {path}")
        root, turn, family, split, start, end = (
            original[field] for field in ("root", "turn", "family", "split", "start", "end")
        )
        observed = set()
        x, y, counts = (original[field] for field in ("x", "y", "visit_count"))
        for i, key in enumerate(zip(root, turn, start, end, strict=True)):
            j = window_index[key]
            window = dataset.windows[j]
            if (key in observed or family[i] != window.turn.family or split[i] != window.turn.split
                    or not np.array_equal(counts[i], (window.end - window.start) * PASS_COUNT)
                    or not np.allclose(x[i], collapse(dataset.x[j]), rtol=1e-4, atol=2e-6)
                    or not np.allclose(y[i], collapse(dataset.y[j]), rtol=1e-4, atol=2e-6)):
                raise ValueError(f"saved demand disagrees with trace: {path} {key}")
            observed.add(key)
            if i % 128 == 0:
                guard.check()
        if observed != window_index.keys():
            raise ValueError(f"incomplete saved windows: {path}")


def prepare(sources, arm, directory, teacher, guard):
    rosters = read_rosters(sources, arm)
    turns = sorted((turn for roster in rosters for turn in roster.values()),
                   key=lambda turn: (turn.root, turn.turn))
    datasets, indices = {}, {}
    frequency = np.zeros((LAYERS, EXPERTS), dtype=np.int64)
    try:
        for h in HORIZONS:
            windows = [Window(turn, start, end) for turn in turns for start, end in turn.bounds(h)]
            offsets = {(w.turn.root, w.turn.turn, w.start, w.end): i for i, w in enumerate(windows)}
            shape = len(windows), LAYERS, PASSES
            x = np.lib.format.open_memmap(directory / f"x{h}.npy", mode="w+",
                                         dtype=np.float32, shape=(*shape, EXPERTS * 2))
            y = np.lib.format.open_memmap(directory / f"y{h}.npy", mode="w+",
                                         dtype=np.float32, shape=(*shape, EXPERTS))
            datasets[h] = Dataset(x, y, windows)
            indices[h] = offsets
        for source, roster in zip(sources, rosters, strict=True):
            arm_dir = source / arm
            for turn, mass, turn_frequency in trace_turns(arm_dir / "trace.jsonl", roster, guard):
                frequency += turn_frequency
                for h, dataset in datasets.items():
                    start, end = next(turn.bounds(h))
                    index = indices[h][turn.root, turn.turn, start, end]
                    write_windows(turn, mass, h, index, dataset)
                guard.check()
            if teacher:
                for h, dataset in datasets.items():
                    shard_windows = {key: i for key, i in indices[h].items() if key[:2] in roster}
                    verify_original(arm_dir / f"demand_H{h}.npz", h, dataset, shard_windows, guard)
        for dataset in datasets.values():
            dataset.x.flush()
            dataset.y.flush()
        return datasets, normalize(frequency)
    except BaseException:
        for dataset in datasets.values():
            dataset.close()
        raise
