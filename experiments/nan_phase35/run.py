"""One memory-bounded pass over immutable STORYAUDIT30 residual captures."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import fmean
import sys
import threading
import time

from experiments.nan_ram28.resources import memory
from experiments.nan_ranksite13.resources import below_normal


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ORIGINAL = REPO / "runs" / "nan_storyaudit30"
AGGREGATE = REPO / "reports/nanbeige_bpd_gym/nan_storyaudit30_results.json"
MANIFEST = ROOT / "SOURCE_MANIFEST.json"
ARTIFACT = Path("E:/codex-artifacts/NanbeigeRam28-20261121/nf4")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    return sha(path.read_bytes())


def read_bound(path, expected):
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f"Immutable artifact digest mismatch: {path.name}")
    return data


def json_bound(path, expected):
    return json.loads(read_bound(path, expected))


def put(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def registration():
    files = [ROOT / name for name in ("registration.json", "run.py", "PROTOCOL.md", ".gitattributes", "__init__.py")]
    files += [REPO / name for name in ("experiments/nan_ram28/resources.py", "experiments/nan_ranksite13/resources.py")]
    return {"files_sha256": {p.relative_to(REPO).as_posix(): digest_file(p) for p in files},
            "data_access": "immutable audited native NF4 story residuals only"}


@contextmanager
def lease(cfg):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(CUDA_VISIBLE_DEVICES="", USE_TORCH="0")
    below_normal()
    start = time.monotonic()
    stop = threading.Event()
    samples = []
    if memory()["available_physical_bytes"] < cfg["minimum_start_available_RAM_bytes"]:
        raise RuntimeError("Phase audit requires 5 GiB available for admission")

    def check():
        sample = memory()
        samples.append(sample)
        if (sample["available_physical_bytes"] < cfg["minimum_available_RAM_bytes"]
                or sample["private_bytes"] > cfg["maximum_private_bytes"]
                or time.monotonic() - start > cfg["maximum_seconds"]):
            raise RuntimeError("Phase audit resource limit exceeded")

    def monitor():
        while not stop.wait(2):
            try:
                check()
            except Exception as error:
                print(error, file=sys.stderr, flush=True)
                os._exit(3)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        check()
        yield check, samples
        check()
    finally:
        stop.set()
        watcher.join()


def ratio(top, bottom):
    return top / bottom if bottom > 0 else None


def cosine(dot, x2, y2):
    if x2 <= 0 or y2 <= 0:
        return None
    return max(-1.0, min(1.0, dot / math.sqrt(x2 * y2)))


def compare(x, y, coordinates, np):
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite residual values")
    xx, yy, xy = float(x @ x), float(y @ y), float(x @ y)
    d = y - x
    d2 = float(d @ d)
    full = cosine(xy, xx, yy)
    result = {"cosine": full, "squared_distance": d2}
    for index in coordinates:
        a, b = float(x[index]), float(y[index])
        removed = cosine(xy - a * b, xx - a * a, yy - b * b)
        result[f"{index}/omitted_cosine_minus_full"] = None if full is None or removed is None else removed - full
        result[f"{index}/delta_energy_share"] = ratio((b - a)**2, d2)
        result[f"{index}/left_energy_share"] = ratio(a * a, xx)
        result[f"{index}/right_energy_share"] = ratio(b * b, yy)
    return result


def observations(cfg, np, check):
    public = json_bound(AGGREGATE, cfg["source_aggregate_sha256"])
    assert public["status"] == "complete_story_block" and public["captured_prefixes"] == cfg["registered_prefixes"]
    audits = json_bound(ORIGINAL / "private_episode_audits.json", cfg["private_audit_sha256"])
    selection = json_bound(ORIGINAL / "selected_closed_entries.json", cfg["selection_sha256"])
    manifest = json.loads((ARTIFACT / "manifest.json").read_bytes())
    assert manifest["source_manifest_sha256"] == cfg["ram28_source_manifest_sha256"]
    assert len(audits) == len(selection) == cfg["registered_trajectories"]
    by_slot = {row["slot"]: row for row in selection}
    coords = [cfg["primary_channel"], *cfg["neighbors"]]
    candidates = [i for i in range(cfg["width"]) if i not in coords]
    coords += random.Random(cfg["random_control_seed"]).sample(candidates, cfg["random_control_count"])
    first = {}
    summaries = []
    coordinate_pairs = {}
    layers = cfg["layers"]
    for entry in audits:
        slot, root, arm, world, split = (entry[k] for k in ("slot", "root", "arm", "world", "split"))
        expected = by_slot[slot]
        assert (entry["closed_file"], entry["closed_sha256"]) == (expected["file"], expected["sha256"])
        assert re.fullmatch(r"baseline_[0-9]{4}_r[01]_closed.json", expected["file"])
        closed = json_bound(ARTIFACT / expected["file"], expected["sha256"])
        assert closed["source_id"] == entry["source_id"] and len(closed["fragments"]) == entry["prefixes"]
        stem = expected["file"].removesuffix("_closed.json")
        series = defaultdict(list)
        pairs = defaultdict(list)
        for step, fragment in enumerate(closed["fragments"]):
            if step % 32 == 0:
                check()
            assert fragment["file"] == f"{stem}_p{step:03d}.json"
            prefix = json_bound(ARTIFACT / fragment["file"], fragment["sha256"])
            assert prefix["step"] == step and prefix["source_id"] == entry["source_id"]
            assert prefix["state_file"] == f"{stem}_p{step:03d}.npz"
            saved_shape = (prefix["batch_size"], prefix["batch_row"], prefix["actual_length"],
                           prefix["padded_length"], prefix["padding_tokens"])
            assert tuple(entry["batch_shapes"][step]) == saved_shape
            wanted = {(item["layer"], item["loop"]): item["key"] for item in prefix["state_index"]
                      if item["site"] == "residual_after_layer" and item["layer"] in layers}
            assert set(wanted) == {(layer, phase) for layer in layers for phase in (1, 2)}
            blob = read_bound(ARTIFACT / prefix["state_file"], prefix["state_sha256"])
            data = {}
            with np.load(BytesIO(blob), allow_pickle=False) as archive:
                for pair, key in wanted.items():
                    value = archive[key]
                    assert value.dtype == np.uint16 and value.shape == (cfg["width"],)
                    data[pair] = (value.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            for layer in layers:
                x, y = data[layer, 1], data[layer, 2]
                values = compare(x, y, coords, np)
                series[layer].append(values)
                pairs[layer].append((x[coords], y[coords]))
            if step == 0:
                twin_key = (root, arm, world)
                assert twin_key not in first
                first[twin_key] = (data, saved_shape)
        for layer in layers:
            coordinate_pairs[slot, layer] = np.asarray(pairs[layer])
            episode = {"root": root, "split": split, "arm": arm, "layer": layer, "slot": slot}
            summaries.append({**episode, "view": "first", "values": series[layer][0],
                              "pair_count": 1, "undefined_full_cosine": int(series[layer][0]["cosine"] is None)})
            values = {metric: mean_or_none(row[metric] for row in series[layer]) for metric in series[layer][0]}
            summaries.append({**episode, "view": "all", "values": values,
                              "pair_count": len(series[layer]),
                              "undefined_full_cosine": sum(row["cosine"] is None for row in series[layer])})
        if slot % 12 == 11:
            print(f"Verified {slot + 1} immutable episodes", flush=True)
    assert sum(len(v) for (slot, layer), v in coordinate_pairs.items() if layer == layers[0]) == cfg["registered_prefixes"]
    assert len({entry["root"] for entry in audits}) == 12
    return summaries, coordinate_pairs, first, coords


def mean_or_none(values):
    finite = [v for v in values if v is not None]
    return fmean(finite) if finite else None


def group(items, fields):
    roots = defaultdict(lambda: defaultdict(list))
    for item in items:
        key = tuple(item[f] for f in fields)
        roots[key][item["root"]].append(item)
    out = []
    for key, separated in sorted(roots.items()):
        metrics = next(iter(separated.values()))[0]["values"]
        means = {}
        support = {}
        for metric in metrics:
            per_root = [mean_or_none(item["values"][metric] for item in values) for values in separated.values()]
            means[metric] = mean_or_none(per_root)
            support[metric] = sum(v is not None for v in per_root)
        observations = [item for items in separated.values() for item in items]
        out.append({**dict(zip(fields, key)), "roots": len(separated), "episode_or_twin_pairs": len(observations),
                    "represented_vector_pairs": sum(item.get("pair_count", 1) for item in observations),
                    "undefined_full_cosine_pairs": sum(item.get("undefined_full_cosine", int(item["values"].get("cosine", 0) is None)) for item in observations),
                    "root_macro": means, "observed_roots": support})
    return out


def phase_thresholds(summaries, coordinate_pairs, coords, layers, np):
    cuts = {}
    result = []
    for layer in layers:
        root_means = defaultdict(list)
        for item in summaries:
            if item["layer"] == layer and item["split"] == "calibration" and item["view"] == "first":
                root_means[item["root"]].append(coordinate_pairs[item["slot"], layer][0])
        assert len(root_means) == 3
        means = np.mean([np.mean(rows, axis=0) for rows in root_means.values()], axis=0)
        midpoint = means.mean(axis=0)
        direction = np.sign(means[1] - means[0])
        cuts[layer] = {"CAL_roots": len(root_means), "phase1_mean": dict(zip(map(str, coords), means[0].tolist())),
                       "phase2_mean": dict(zip(map(str, coords), means[1].tolist())),
                       "cut": dict(zip(map(str, coords), midpoint.tolist())),
                       "phase2_direction": dict(zip(map(str, coords), direction.tolist()))}
        for item in summaries:
            if item["layer"] != layer:
                continue
            states = coordinate_pairs[item["slot"], layer]
            if item["view"] == "first":
                states = states[:1]
            signed = (states - midpoint) * direction
            balanced = ((signed[:, 0] <= 0).mean(axis=0) + (signed[:, 1] > 0).mean(axis=0)) * 0.5
            result.append({**item, "values": dict(zip(map(str, coords), balanced.tolist()))})
    return cuts, group(result, ("split", "arm", "layer", "view"))


def twins(first, coords, layers, np):
    shape_counts = defaultdict(int)
    pairs = []
    for (root, arm, world), (left, left_shape) in first.items():
        if world != "A":
            assert world == "B"
            continue
        right, right_shape = first[root, arm, "B"]
        split = root.split(":")[1]
        shape_counts[split, arm, "total_AB"] += 1
        if left_shape[2] != right_shape[2]:
            shape_counts[split, arm, "unequal_semantic_length"] += 1
        if any(left_shape[i] != right_shape[i] for i in (0, 3, 4)):
            shape_counts[split, arm, "unequal_batch_canvas"] += 1
        for layer in layers:
            for phase in (1, 2):
                values = compare(left[layer, phase], right[layer, phase], coords, np)
                pairs.append({"root": root, "split": split, "arm": arm, "layer": layer,
                              "phase": phase, "values": values})
    assert len(pairs) == 36 * len(layers) * 2
    return group(pairs, ("split", "arm", "layer", "phase")), [
        {"split": key[0], "arm": key[1], "metric": key[2], "count": value} for key, value in sorted(shape_counts.items())]


def main():
    if not __debug__:
        raise RuntimeError("Run the geometry audit with assertions enabled")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.freeze:
        put(MANIFEST, registration())
        print(sha(MANIFEST.read_bytes()))
        return
    assert args.output and json.loads(MANIFEST.read_bytes()) == registration()
    cfg = json.loads((ROOT / "registration.json").read_bytes())
    with lease(cfg) as (check, samples):
        if version("numpy") != cfg["numpy_version"]:
            raise ValueError("Wrong pinned NumPy environment")
        import numpy as np

        summary, pair_values, first, coords = observations(cfg, np, check)
        cuts, phase_scores = phase_thresholds(summary, pair_values, coords, cfg["layers"], np)
        twin_metrics, shapes = twins(first, coords, cfg["layers"], np)
        assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
        out = {
            "experiment": cfg["experiment"], "status": "complete", "roots": {"CAL": 3, "DEV": 9},
            "prefixes": cfg["registered_prefixes"], "trajectories": cfg["registered_trajectories"],
            "layers": cfg["layers"], "primary_coordinate": cfg["primary_channel"], "registered_coordinates": coords,
            "coordinate_geometry": group(summary, ("split", "arm", "layer", "view")),
            "CAL_thresholds": cuts, "phase_balanced_accuracy": phase_scores,
            "first_prefix_twin_geometry": twin_metrics, "first_prefix_twin_canvas_counts": shapes,
            "source_manifest_sha256": digest_file(MANIFEST), "STORYAUDIT30_result_sha256": cfg["source_aggregate_sha256"],
            "original_target_status": "pending_canonical_RAM28_receipt", "new_pretrained_forwards": 0,
            "formal_comparisons": 0, "sealed_test_accessed": False, "decision_scope": "REDESIGN_OR_PILOT_ONLY",
        }
    out["resources"] = {
        "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
        "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples),
        "samples": len(samples),
    }
    put(args.output, out)
    print(json.dumps({key: out[key] for key in ("status", "roots", "prefixes", "trajectories", "resources")}), flush=True)


if __name__ == "__main__":
    main()
