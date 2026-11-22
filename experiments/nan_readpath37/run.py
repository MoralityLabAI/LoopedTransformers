"""Measured COMP14 direct readout and a PercInterp path-selection negative control."""
import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
from importlib.metadata import version
import json
import math
from pathlib import Path
from statistics import fmean, median
import sys

from experiments.nan_phase35.run import lease


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
LOCAL = REPO / "runs/nan_comp14"
FIT = REPO / "runs/nan_fit16_io18"
SHARD = "model-00001-of-00002.safetensors"


def sha(data):
    return sha256(data).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream, check=lambda: None):
    value = sha256()
    for index, block in enumerate(iter(lambda: stream.read(2**20), b"")):
        if index % 64 == 0:
            check()
        value.update(block)
    return value.hexdigest()


def bound(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError(f"Source hash mismatch: {path.name}")
    return raw


def write_new(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def seal():
    paths = [HERE / name for name in ("__init__.py", ".gitattributes", "PROTOCOL.md", "registration.json", "run.py")]
    paths += [REPO / name for name in ("experiments/nan_phase35/run.py", "experiments/nan_ram28/resources.py",
                                      "experiments/nan_ranksite13/resources.py")]
    return {"experiment": "NAN-READPATH-37",
            "files_sha256": {path.relative_to(REPO).as_posix(): digest(path) for path in paths},
            "stage": "prospective_cpu_only_comp14_readout"}


def bf16(x, np):
    """Round float32 to BF16 nearest even and retain it in a float32 array."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def decode_bf16(data, np):
    return (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


class FixedHead:
    def __init__(self, stream, snapshot, cfg, np, check):
        if sha256_stream(stream, check) != cfg["snapshot_pins"][SHARD]:
            raise ValueError("Checkpoint head shard does not match its lock")
        stream.seek(0)
        count = int.from_bytes(stream.read(8), "little")
        if not 0 < count < 2**20:
            raise ValueError("Invalid safetensors header size")
        weights = json.loads(stream.read(count))
        index = json.loads(bound(snapshot / "model.safetensors.index.json",
                                 cfg["snapshot_pins"]["model.safetensors.index.json"]))["weight_map"]
        config = json.loads(bound(snapshot / "config.json", cfg["snapshot_pins"]["config.json"]))
        bound(snapshot / "modeling_nanbeige.py", cfg["snapshot_pins"]["modeling_nanbeige.py"])
        self.stream, self.np = stream, np
        self.width = cfg["width"]
        self.vocab = config["vocab_size"]
        self.eps = np.float32(config["rms_norm_eps"])
        self.base = 8 + count
        self.cache = {}
        for name, shape in [("lm_head.weight", [self.vocab, self.width]), ("model.norm.weight", [self.width])]:
            item = weights[name]
            assert index[name] == SHARD and item["dtype"] == "BF16" and item["shape"] == shape
            assert item["data_offsets"][1] - item["data_offsets"][0] == 2 * math.prod(shape)
        self.weight_offset = weights["lm_head.weight"]["data_offsets"][0]
        self.gamma = self.read(weights["model.norm.weight"]["data_offsets"][0])

    def read(self, offset):
        self.stream.seek(self.base + offset)
        raw = self.stream.read(2 * self.width)
        if len(raw) != 2 * self.width:
            raise ValueError("Incomplete pinned head bytes")
        result = decode_bf16(raw, self.np)
        assert self.np.isfinite(result).all()
        return result

    def row(self, token):
        assert 0 <= token < self.vocab
        if token not in self.cache:
            self.cache[token] = self.read(self.weight_offset + 2 * self.width * token)
        return self.cache[token]

    def norm(self, states):
        np = self.np
        s = np.asarray(states, dtype=np.float32)
        variance = np.mean(s * s, axis=-1, keepdims=True, dtype=np.float32)
        return bf16(bf16(s * np.reciprocal(np.sqrt(variance + self.eps)), np) * self.gamma, np)

    def logits(self, states, token_ids):
        np = self.np
        weights = np.asarray([self.row(i) for i in token_ids], dtype=np.float64)
        hidden = self.norm(states).astype(np.float64)
        return bf16(hidden @ weights.T, np)


def ranked_entries(weights):
    live = [abs(w) for w in weights if w]
    return [sum(v >= abs(w) for v in live) / len(live) if w else math.inf for w in weights]


def path_members(sources, sinks, topological_nodes, edges, cutoff):
    kept = [(a, b) for a, b, p in edges if p <= cutoff and math.isfinite(p)]
    reached, reaches_output = set(sources), set(sinks)
    for node in topological_nodes:
        if any(b == node and a in reached for a, b in kept):
            reached.add(node)
    for node in reversed(topological_nodes):
        if any(a == node and b in reaches_output for a, b in kept):
            reaches_output.add(node)
    return reached & reaches_output, kept


def direct_path_control(source_weights, margin_effects):
    rank = len(source_weights)
    nodes = list(range(rank + 2))
    interior = set(nodes[1:-1])
    end = nodes[-1]
    edges = [(0, i, key) for i, key in enumerate(ranked_entries(source_weights), 1)]
    edges += [(i, end, key) for i, key in enumerate(ranked_entries(margin_effects), 1)]
    cutoffs = sorted({0.0, 1.0, *(p for _, _, p in edges if math.isfinite(p))})
    nonempty = 0
    for cutoff in cutoffs:
        path, kept = path_members({0}, {end}, nodes, edges, cutoff)
        degree = {b for _, b in kept} & {a for a, _ in kept}
        if path & interior != degree & interior:
            raise AssertionError("Shallow path/degree identity failed")
        nonempty += bool(path & interior)
    return len(cutoffs), nonempty


def self_test(np):
    np.testing.assert_array_equal(bf16(np.array([1.00390625, 1.01171875], np.float32), np),
                                  np.array([1.0, 1.015625], np.float32))
    assert ranked_entries([2.0, -2.0, 1.0, 0.0]) == [2 / 3, 2 / 3, 1, math.inf]
    assert direct_path_control([1, 2, 0], [-3, 1, 0]) == (3, 1)
    path, kept = path_members({0}, {5}, range(6), [(0, 1, 0.5), (1, 2, 0.5), (3, 4, 0.5), (4, 5, 0.5)], 1.0)
    assert not path and ({b for _, b in kept} & {a for a, _ in kept}) == {1, 4}


def measure(head, h, directions, requested, target, foil, cfg, np):
    h64 = h.astype(np.float64)
    norm = float(np.linalg.norm(h64))
    scale = requested * norm
    edited = [bf16(h64 + sign * scale * directions, np) for sign in (1, -1)]
    reached = [np.linalg.norm(values.astype(np.float64) - h64, axis=1) for values in edited]
    valid = np.logical_and(*(abs(values / scale - 1) <= cfg["dose_relative_tolerance"] for values in reached))
    shifted = [head.logits(values, [target, foil]).astype(np.float64) for values in edited]
    margin = [scores[:, 0] - scores[:, 1] for scores in shifted]
    central = (margin[0] - margin[1]) / (2 * requested)
    return [float(v) if keep else None for v, keep in zip(central, valid)], [
        float(np.max(abs(values / scale - 1))) for values in reached
    ]


def joined(cfg):
    pins = cfg["common_pins"]
    for name in pins:
        bound(REPO / name, pins[name])
    pre = "experiments/nan_comp14/fixtures/"
    rows = {}
    for kind in ("public", "host_labels"):
        name = pre + kind + ".jsonl"
        rows[kind] = [json.loads(line) for line in bound(REPO / name, pins[name]).splitlines()]
    assert len(rows["public"]) == len(rows["host_labels"]) == 240
    selected = {}
    for slot, (record, label) in enumerate(zip(rows["public"], rows["host_labels"])):
        assert all(record[key] == label[key] for key in ("id", "root", "split", "world", "arm"))
        if label["arm"] in cfg["arms"]:
            selected[slot] = label
    assert len(selected) == 144
    by_world = {(row["root"], row["arm"], row["world"]): row["answer"] for row in selected.values()}
    assert len(by_world) == len(selected)
    return selected, by_world


def case_states(case, payload, np):
    index = {(item["loop"], item["site"]): item["array_key"] for item in case["state_index"]
             if item["layer_zero_based"] == 21}
    use = ("residual_after_layer", "down_proj_input")
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        result = {(p, site): archive[index[p, site]].astype(np.float32) for p in (1, 2) for site in use}
    assert all(np.isfinite(a).all() and a.ndim == 1 for a in result.values())
    return result


def backend_rows(backend, head, label_ids, cfg, selected, by_world, np, check):
    pin = cfg["backends"][backend]
    location = LOCAL / backend
    audit = json.loads(bound(LOCAL / "audits" / f"{backend}.json", pin["audit"]))
    manifest = json.loads(bound(location / "manifest.json", pin["manifest"]))
    factor_dir = FIT / backend / "t2_layer21_down_proj_rank8"
    factor_receipt = json.loads(bound(factor_dir / "receipt.json", pin["factor_receipt"]))
    assert len(audit["records"]) == audit["contexts_checked"] == audit["registered_contexts"] == 240
    assert audit["missing_contexts"] == 0 and audit["worker_status"] == "completed"
    assert audit["status"] == "passed"
    assert manifest["target_parameter_sha256"] == factor_receipt["target_parameter_sha256_unchanged"] == pin["target"]
    assert manifest["checkpoint_revision"] == cfg["revision"] and factor_receipt["optimizer_steps_completed"] == 16

    with np.load(BytesIO(bound(factor_dir / "final.npz", pin["factor_bank"])), allow_pickle=False) as bank:
        V, U = bank["V"].astype(np.float64), bank["U"].astype(np.float64)
    assert V.shape == (cfg["input_width"], cfg["rank"]) and U.shape == (cfg["rank"], cfg["width"])
    assert np.isfinite(V).all() and np.isfinite(U).all()
    norms = np.linalg.norm(U, axis=1)
    assert (norms > 0).all()
    random_basis = np.linalg.qr(np.random.default_rng(cfg["random_seed"]).normal(
        size=(cfg["width"], cfg["rank"])), mode="reduced")[0].T
    coordinates = np.zeros((3, cfg["width"]), dtype=np.float64)
    coordinates[np.arange(3), [2278, 2279, 2280]] = 1
    records = []
    for slot, source in selected.items():
        check()
        verified = audit["records"][slot]
        assert all(verified[k] == source[k] for k in ("id", "root", "split", "world", "arm"))
        case = json.loads(bound(location / f"case_{slot:04d}.json", verified["case_sha256"]))
        assert case["id"] == source["id"] and case["state_file_sha256"] == verified["state_sha256"]
        assert case["state_file"] == f"case_{slot:04d}_states.npz"
        states = case_states(case, bound(location / case["state_file"], verified["state_sha256"]), np)
        opposite = "B" if source["world"] == "A" else "A"
        target = label_ids[source["answer"]]
        foil = label_ids[by_world[source["root"], source["arm"], opposite]]
        assert target != foil
        for phase in (1, 2):
            h, x = [states[phase, site] for site in ("residual_after_layer", "down_proj_input")]
            assert h.shape == (cfg["width"],) and x.shape == (cfg["input_width"],)
            stored = case["initial_phases"][phase - 1]
            assert stored["loop"] == phase
            native = {int(k): v for k, v in stored["score"]["selected_logits"].items()}
            native[stored["score"]["free_top_token_id"]] = stored["score"]["free_top_token_logit"]
            replay = head.logits(h, list(native))
            discrepancy = float(np.max(abs(replay - list(native.values()))))
            correct = verified["first_token_correct"]
            native_correct = bool(correct[str(phase)]) if isinstance(correct, dict) else bool(correct[phase - 1])
            info = {k: source[k] for k in ("root", "world", "arm", "split")}
            row = {**info, "phase": phase, "prompt_tokens": len(case["initial_prompt_ids"]),
                   "native_unrestricted_correct": native_correct, "native_top_token_id": stored["score"]["free_top_token_id"],
                   "native_closure_abs_error": discrepancy,
                   "native_closure": discrepancy <= cfg["native_logit_abs_tol"]}
            records.append(row)
            if not row["native_closure"]:
                continue
            raw_margin = head.logits(h, [target, foil]).astype(np.float64)
            row["replayed_bare_label_margin"] = float(raw_margin[0] - raw_margin[1])
            act = x.astype(np.float64) @ V
            physical = abs(act) * norms
            orientation = np.where(act < 0, -1, 1)
            writes = orientation[:, None] * U / norms[:, None]
            directions = np.concatenate((writes, random_basis, coordinates))
            for dose in cfg["dose_fractions"]:
                scores, relative_errors = measure(head, h, directions, dose, target, foil, cfg, np)
                row[str(dose)] = {"signed": scores, "max_relative_dose_error": relative_errors}
            signed_path = row[str(cfg["dose_fractions"][-1])]["signed"][:cfg["rank"]]
            row["signed_direct_graph"] = list(zip(physical.tolist(), signed_path))
            if all(v is not None for v in signed_path):
                row["path_identity_cutoffs"], row["path_nonempty_cutoffs"] = direct_path_control(
                    physical.tolist(), signed_path)
    assert len(records) == 288
    return records


def summarize(records, cfg):
    cells = defaultdict(list)
    for row in records:
        cells[row["split"], row["arm"], row["phase"]].append(row)
    aggregate = []
    for (split, arm, phase), group in sorted(cells.items()):
        twins = defaultdict(list)
        for row in group:
            twins[row["root"]].append(row)
        assert len(twins) == cfg["roots"][split] and all(len(rows) == 2 for rows in twins.values())
        good = [row for row in group if row["native_closure"]]
        complete = [pair for pair in twins.values() if all(r["native_closure"] for r in pair)]
        out = {
            "split": split, "arm": arm, "phase": phase, "roots": len(twins), "worlds": len(group),
            "same_B1_prompt_length_twins": sum(a["prompt_tokens"] == b["prompt_tokens"] for a, b in twins.values()),
            "max_native_logit_closure_error": max(r["native_closure_abs_error"] for r in group),
            "passed_native_closure": len(good),
            "native_unrestricted_both_correct_roots": sum(all(r["native_unrestricted_correct"] for r in p) for p in twins.values()),
            "bare_label_margin_positive_both_worlds": sum(all(r["replayed_bare_label_margin"] > 0 for r in p) for p in complete),
            "mean_root_bare_label_margin": fmean(fmean(r["replayed_bare_label_margin"] for r in p) for p in complete) if complete else None,
            "complete_twins": len(complete), "graphs_checked": sum("path_identity_cutoffs" in r for r in good),
            "path_degree_identity_cutoffs": sum(r.get("path_identity_cutoffs", 0) for r in good),
            "nonempty_identity_cutoffs": sum(r.get("path_nonempty_cutoffs", 0) for r in good),
        }
        names = ["IO18_T2", "orthonormal_random", "channel_2278", "channel_2279", "channel_2280"]
        rank = cfg["rank"]
        spans = [(0, rank), (rank, 2 * rank), (2 * rank, 2 * rank + 1), (2 * rank + 1, 2 * rank + 2),
                 (2 * rank + 2, 2 * rank + 3)]
        for dose in map(str, cfg["dose_fractions"]):
            directions = {}
            for name, (lo, hi) in zip(names, spans):
                values = [v for row in good for v in row[dose]["signed"][lo:hi] if v is not None]
                # Every complete root is averaged independently over its two worlds and all complete directions.
                complete_vectors = [
                    [v for row in pair for v in row[dose]["signed"][lo:hi]]
                    for pair in complete
                ]
                root_values = [v for v in complete_vectors if all(x is not None for x in v)]
                directions[name] = {"qualified_world_direction_pairs": len(values),
                                    "registered_world_direction_pairs": len(group) * (hi - lo),
                                    "complete_roots": len(root_values),
                                    "root_macro_abs_response": fmean(fmean(map(abs, v)) for v in root_values) if root_values else None,
                                    "median_abs_response_descriptive": median(map(abs, values)) if values else None}
            matched = []
            for pair in complete:
                vectors = [row[dose]["signed"][:2 * rank] for row in pair]
                if all(all(value is not None for value in vector) for vector in vectors):
                    learned = fmean(fmean(map(abs, vector[:rank])) for vector in vectors)
                    sham = fmean(fmean(map(abs, vector[rank:])) for vector in vectors)
                    matched.append((learned, sham))
            directions["matched_learned_vs_random"] = {
                "complete_roots": len(matched),
                "mean_abs_learned": fmean(a for a, _ in matched) if matched else None,
                "mean_abs_random": fmean(b for _, b in matched) if matched else None,
                "root_macro_difference": fmean(a-b for a, b in matched) if matched else None,
            }
            out[dose] = directions
        aggregate.append(out)
    return aggregate


def main():
    if not __debug__:
        raise RuntimeError("Run with verification assertions enabled")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(MANIFEST, seal())
        print(digest(MANIFEST))
        return
    assert args.snapshot and args.output and json.loads(MANIFEST.read_bytes()) == seal()
    cfg = json.loads((HERE / "registration.json").read_bytes())
    assert args.snapshot.name == cfg["revision"]
    with lease(cfg) as (check, samples):
        assert version("numpy") == "2.5.5" and version("tokenizers") == "0.20.3"
        import numpy as np
        from tokenizers import Tokenizer

        self_test(np)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        bound(args.snapshot / "tokenizer.json", cfg["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        label_tokens = {label: tok.encode(label, add_special_tokens=False).ids
                        for label in "ABCDEFGHIJKLMNOP"}
        assert all(len(v) == 1 for v in label_tokens.values()) and len({v[0] for v in label_tokens.values()}) == 16
        labels = {label: v[0] for label, v in label_tokens.items()}
        selected, by_world = joined(cfg)
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, cfg, np, check)
            results = {}
            for backend in cfg["backends"]:
                check()
                records = backend_rows(backend, head, labels, cfg, selected, by_world, np, check)
                write_new(args.output.with_name(f"{backend}_private.json"), records)
                results[backend] = summarize(records, cfg)
                print(f"Verified {backend}: {len(records)} phase states", flush=True)
        assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
        check()
    out = {
        "experiment": cfg["experiment"], "status": "completed", "source_manifest_sha256": digest(MANIFEST),
        "roots": cfg["roots"], "states_per_backend": 288, "new_pretrained_forwards": 0,
        "formal_comparisons": 0, "sealed_test_accessed": False, "physical_relative_doses": cfg["dose_fractions"],
        "per_backend": results, "resource": {
            "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
            "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples),
            "samples": len(samples),
        },
        "verdict": "direct_readout_and_selector_negative_control_only",
    }
    write_new(args.output, out)
    print(json.dumps({k: out[k] for k in ("status", "states_per_backend", "resource")}), flush=True)


if __name__ == "__main__":
    main()
