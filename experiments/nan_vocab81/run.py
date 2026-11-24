"""Audit the full vocabulary on integrity-closed COMP14 recurrent states."""

import argparse
from collections import defaultdict
import json
from importlib.metadata import version
from math import isclose
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_comp14.parser import label
from experiments.nan_lens40.run import ARCHIVE, read_cases
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import (
    FixedHead, SHARD, bf16, bound, decode_bf16, digest, joined, write_new,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
BASE = REPO / "experiments/nan_readpath37/registration.json"
INVERSE = REPO / "reports/nanbeige_bpd_gym/nan_inverse65_results.json"
LABELS = "ABCDEFGHIJKLMNOP"
INVALID = len(LABELS)
STAGES = tuple((p, l) for p in (1, 2) for l in (17, 21))
CAPTURED = tuple((p, l) for p in (1, 2) for l in (5, 11, 17, 21))
PRIMARY = (2, 17)
KEYS = ("id", "root", "split", "arm", "world")
TILE = 768
BUDGET = dict(minimum_start_available_RAM_bytes=5 * 2**30,
              minimum_available_RAM_bytes=4 * 2**30, maximum_private_bytes=768 * 2**20,
              maximum_seconds=600)


def seal():
    paths = [HERE / n for n in (".gitattributes", "__init__.py", "PROTOCOL.md", "run.py")]
    paths += [BASE, INVERSE, *[REPO / n for n in (
        "experiments/nan_comp14/parser.py", "experiments/nan_lens40/run.py",
        "experiments/nan_readpath37/run.py", "experiments/nan_lens40/SOURCE_MANIFEST.json",
        "experiments/nan_inverse65/SOURCE_MANIFEST.json", "experiments/nan_phase35/run.py",
        "experiments/nan_ram28/resources.py", "experiments/nan_ranksite13/resources.py",
    )]]
    return {"experiment": "NAN-VOCAB-81", "resources": BUDGET, "tile_rows": TILE,
            "files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in paths}}


def token_classes(tok, count, np, check):
    classes = []
    for start in range(0, count, 4096):
        check()
        texts = tok.decode_batch([[i] for i in range(start, min(start + 4096, count))],
                                 skip_special_tokens=False)
        classes.extend(INVALID if name is None else LABELS.index(name)
                       for name in map(label, texts))
    return np.asarray(classes, dtype=np.int8)


def reduce_tiles(blocks, classes, canonical, size, np):
    peak = np.full((size, INVALID + 1), -np.inf)
    token = np.full(peak.shape, -1, dtype=np.int32)
    bare = np.full((size, INVALID), -np.inf)
    logz = np.full(size, -np.inf)
    seen = 0
    rows = np.arange(size)
    for block in blocks:
        end = seen + block.shape[1]
        if block.shape[0] != size or end > len(classes) or not np.isfinite(block).all():
            raise ValueError("Invalid vocabulary score tile")
        logz = np.logaddexp(logz, np.logaddexp.reduce(block, axis=1))
        for c in np.unique(classes[seen:end]):
            indices = np.flatnonzero(classes[seen:end] == c)
            subset = block[:, indices]
            winner = indices[subset.argmax(axis=1)]
            values = block[rows, winner]
            improves = values > peak[:, c]
            peak[improves, c] = values[improves]
            token[improves, c] = seen + winner[improves]
        for c, idx in enumerate(canonical):
            if seen <= idx < end:
                bare[:, c] = block[:, idx - seen]
        seen = end
    if seen != len(classes) or not np.isfinite(peak).all() or not np.isfinite(bare).all():
        raise ValueError("Full vocabulary or label support did not close")
    highest = peak.max(axis=1)
    top = np.where(peak == highest[:, None], token, len(classes)).min(axis=1)
    return peak, bare, highest, top, logz


def fixture(np):
    classes = np.array([*range(INVALID + 1), 0, INVALID, 2])
    canonical = list(range(INVALID))
    sample = np.stack([np.arange(20) % 7, np.full(20, -3.), np.arange(20)[::-1]])
    tiled = reduce_tiles((sample[:, :6], sample[:, 6:13], sample[:, 13:]),
                         classes, canonical, 3, np)
    np.testing.assert_array_equal(tiled[0], np.stack(
        [sample[:, classes == c].max(axis=1) for c in range(INVALID + 1)], axis=1))
    np.testing.assert_array_equal(tiled[1], sample[:, canonical])
    np.testing.assert_array_equal(tiled[2], sample.max(axis=1))
    np.testing.assert_array_equal(tiled[3], sample.argmax(axis=1))
    np.testing.assert_allclose(tiled[4], np.log(np.exp(sample).sum(axis=1)), atol=1e-12, rtol=0)
    return {"dense_reference": True, "same_class_alias": True, "cross_class_tie": True}


def stream_tiles(head, states, check, np):
    normalized = head.norm(states).astype(np.float64)
    head.stream.seek(head.base + head.weight_offset)
    for start in range(0, head.vocab, TILE):
        check()
        count = min(TILE, head.vocab - start)
        raw = head.stream.read(2 * head.width * count)
        if len(raw) != 2 * head.width * count:
            raise ValueError("Truncated head tensor")
        weights = decode_bf16(raw, np).reshape(count, head.width).astype(np.float64)
        yield bf16(normalized @ weights.T, np).astype(np.float64)


def native_controls(backend, base, selected, tok, check, np):
    cfg = base["backends"][backend]
    audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", cfg["audit"]))
    native = []
    for slot, source in selected.items():
        check()
        trusted = audit["records"][slot]
        path = ARCHIVE / backend / f"case_{slot:04d}.json"
        case = json.loads(bound(path, trusted["case_sha256"]))
        if any(case[k] != source[k] or trusted[k] != source[k] for k in KEYS):
            raise ValueError("Native boundary roster differs")
        phases = []
        if [entry["loop"] for entry in case["initial_phases"]] != [1, 2]:
            raise ValueError("Native boundary pass order differs")
        for entry in case["initial_phases"]:
            score = entry["score"]
            top = score["free_top_token_id"]
            decoded = tok.decode([top], skip_special_tokens=False)
            if decoded != entry["token_text"]:
                raise ValueError("Native token decoding differs")
            phases.append((top, score["free_top_token_logit"], score["full_log_normalizer"],
                           label(decoded) == source["answer"]))
        if [int(row[3]) for row in phases] != trusted["first_token_correct"]:
            raise ValueError("Native parser disagrees with independent audit")
        native.append(phases)
    return np.asarray(native)


def group_indices(rows, base):
    groups = defaultdict(lambda: defaultdict(dict))
    for i, row in enumerate(rows):
        roots = groups[row["split"], row["arm"]]
        pair = roots[row["root"]]
        if row["world"] in pair:
            raise ValueError("Duplicate archived world")
        pair[row["world"]] = i
    if {key: len(roots) for key, roots in groups.items()} != {
        (split, arm): count for split, count in base["roots"].items() for arm in base["arms"]
    }:
        raise ValueError("Incomplete root roster")
    return {k: [tuple(pair[w] for w in ("A", "B")) for pair in roots.values()]
            for k, roots in groups.items()}


def margin_details(peak, bare, target, alternate, np):
    indices = np.arange(len(target))
    valid = peak[:, :INVALID].copy()
    valid[indices, target] = -np.inf
    competing_label = valid.max(axis=1)
    invalid = peak[:, INVALID]
    best_gold = peak[indices, target]
    bare_competitors = bare.copy()
    bare_competitors[indices, target] = -np.inf
    bare_gold = bare[indices, target]
    return np.stack((bare_gold - bare[indices, alternate], bare_gold - bare_competitors.max(axis=1),
                     best_gold - competing_label, best_gold - np.maximum(competing_label, invalid),
                     best_gold - invalid, competing_label - invalid), axis=1)


def describe(indices, pairs, margins, prediction, native, np):
    labels = ("pair_bare", "alphabet_bare", "valid_spellings", "all_tokens", "against_invalid")
    result = {}
    for j, name in enumerate(labels):
        x = margins[indices, j]
        result[name] = {"strict_worlds": int((x > 0).sum()), "ties": int((x == 0).sum()),
                        "both_world_roots": sum(bool((margins[list(pair), j] > 0).all()) for pair in pairs),
                        "mean_margin": fmean(x)}
    top = margins[indices, 3] <= 0
    kind = margins[indices, 5]
    result["nonwins_by_strongest_competitor"] = {
        "invalid": int((top & (kind < 0)).sum()), "other_label": int((top & (kind > 0)).sum()),
        "tied_competitor_classes": int((top & (kind == 0)).sum()),
    }
    result["deterministic_token_correct"] = int(prediction[indices].sum())
    categories = ("both_wrong", "repair", "regression", "both_correct")
    verdicts = {}
    for j, name in enumerate(categories):
        is_category = (2 * native[indices, 0] + native[indices, 1]) == j
        verdicts[name] = {"worlds": int(is_category.sum()),
                          "pair_bare_wins": int((is_category & (margins[indices, 0] > 0)).sum()),
                          "all_token_wins": int((is_category & (margins[indices, 3] > 0)).sum())}
    result["by_native_trajectory"] = verdicts
    return result


def run(base, inverse, snapshot, tok, work, check, np):
    selected, _ = joined(base)
    canonical = [tok.encode(c, add_special_tokens=False).ids for c in LABELS]
    if any(len(ids) != 1 for ids in canonical):
        raise ValueError("A canonical bare label is not a singleton")
    canonical = [ids[0] for ids in canonical]
    records = {}
    with (snapshot / SHARD).open("rb") as stream:
        head = FixedHead(stream, snapshot, base, np, check)
        classes = token_classes(tok, head.vocab, np, check)
        if classes[canonical].tolist() != list(range(INVALID)):
            raise ValueError("A canonical token fails its parser class")
        for backend in base["backends"]:
            rows, captured, selected_error = read_cases(
                backend, base, selected, CAPTURED, canonical, head, check, np)
            actual = native_controls(backend, base, selected, tok, check, np)
            groups = group_indices(rows, base)
            target = np.asarray([LABELS.index(row["answer"]) for row in rows])
            alternate = np.full(len(rows), -1)
            for pairs in groups.values():
                for a, b in pairs:
                    alternate[a], alternate[b] = target[b], target[a]
            if (alternate < 0).any() or (alternate == target).any():
                raise ValueError("A counterfactual pair lacks distinct target labels")
            native = actual[:, :, 3].astype(np.int8)
            sites, private, boundary = {}, {}, {}
            for phase, layer in STAGES:
                stage = f"{phase}/{layer}"
                erased = [pair for key, pairs in groups.items() if key[1] == "focused_erased"
                          for pair in pairs]
                states = captured[phase, layer]
                if any(not np.array_equal(states[a], states[b]) for a, b in erased):
                    raise ValueError("Erased twins differ in the captured physical residual")
                peak, bare, highest, top, logz = reduce_tiles(
                    stream_tiles(head, states, check, np), classes, canonical, len(rows), np)
                expected = head.logits(states, canonical)
                if not np.array_equal(bare, expected):
                    raise ValueError("Streamed canonical rows disagree with restricted-head replay")
                if layer == 21:
                    p = phase - 1
                    top_error = float(np.max(abs(highest - actual[:, p, 1])))
                    logz_error = float(np.max(abs(logz - actual[:, p, 2])))
                    if ((top != actual[:, p, 0]).any() or not np.array_equal(
                            classes[top] == target, native[:, p]) or
                            max(top_error, logz_error) > base["native_logit_abs_tol"]):
                        differences = int((top != actual[:, p, 0]).sum())
                        raise ValueError(f"Native head closure failed at {backend} {stage}: "
                                         f"{differences} IDs, peak {top_error}, logZ {logz_error}")
                    boundary[stage] = {"matching_native_ids": len(top), "peak_max_abs_error": top_error,
                                       "log_normalizer_max_abs_error": logz_error}
                margins = margin_details(peak, bare, target, alternate, np)
                prediction = classes[top] == target
                private[stage] = {"margins": margins.tolist(), "top_ids": top.tolist(),
                                  "top_correct": prediction.tolist()}
                cells = {}
                for key, pairs in groups.items():
                    name = "/".join(key)
                    idx = [i for pair in pairs for i in pair]
                    stats = describe(idx, pairs, margins, prediction, native, np)
                    ref = inverse["per_backend"][backend]["groups"][name]["sites"][f"pass{phase}/layer{layer}"]
                    pair_stats = stats["pair_bare"]
                    if (pair_stats["both_world_roots"] != ref["both_bare_labels_win_roots"]
                            or not isclose(2 * pair_stats["mean_margin"], ref["mean_signed"]["finite_margin"],
                                           abs_tol=1e-10, rel_tol=0)):
                        raise ValueError("Full-vocabulary replay disagrees with INVERSE65")
                    cells[name] = {"roots": len(pairs), "worlds": len(idx)} | stats
                sites[stage] = cells
                print(backend, stage, "verified", flush=True)
            write_new(work / f"{backend}_private.json", {"rows": rows, "site_details": private})
            records[backend] = {"sites": sites, "native_boundary_closure": boundary,
                                "native_selected_max_abs_error": selected_error}
    return {"vocab_rows": head.vocab, "token_class_counts":
            {c: int((classes == i).sum()) for i, c in enumerate(LABELS + "?")},
            "per_backend": records}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--work", type=Path)
    args = cli.parse_args()
    frozen = seal()
    if args.freeze:
        write_new(MANIFEST, frozen)
        return
    if not __debug__ or not args.snapshot or not args.work or json.loads(MANIFEST.read_bytes()) != frozen:
        cli.error("Require enabled assertions, a verified manifest, snapshot and fresh output")
    base = json.loads(BASE.read_bytes())
    if args.snapshot.name != base["revision"]:
        cli.error("Wrong checkpoint revision")
    inverse = json.loads(INVERSE.read_bytes())
    with lease(BUDGET) as (check, samples):
        if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise RuntimeError("Unpinned CPU environment")
        import numpy as np
        from tokenizers import Tokenizer

        fixture_result = fixture(np)
        path = args.snapshot / "tokenizer.json"
        bound(path, base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(path))
        args.work.mkdir(parents=True, exist_ok=False)
        results = run(base, inverse, args.snapshot, tok, args.work, check, np)
        if "torch" in sys.modules or seal() != frozen:
            raise RuntimeError("CPU replay environment or frozen source changed")
    write_new(args.work / "aggregate.json", results | {
        "experiment": frozen["experiment"], "source_manifest_sha256": digest(MANIFEST),
        "unit": "COMP14 pointer graph root", "roots": base["roots"], "worlds_per_backend": 144,
        "primary_site": f"{PRIMARY[0]}/{PRIMARY[1]}", "evidence": "descriptive observational",
        "pretrained_decoder_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "fixture": fixture_result, "status": "passed",
        "resources": {"max_sampled_private_bytes": max(x["private_bytes"] for x in samples),
                      "max_sampled_working_set_bytes": max(x["working_set_bytes"] for x in samples),
                      "min_sampled_available_RAM_bytes": min(x["available_physical_bytes"] for x in samples)},
    })


if __name__ == "__main__":
    main()
