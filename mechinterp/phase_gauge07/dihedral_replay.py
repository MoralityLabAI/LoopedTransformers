"""COE-07R: exact rotation/reversal-invariant telemetry audit.

Reuses the original pinned LoopScope-02 *truth-conditioned correctness* series.
The descriptor requires target labels to form the bit at each visit. These
numbers are diagnostic oracles, not a readout implementable from activations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

try:
    from .score import STRATA, rotate, necklace, select_manifest
    from .telemetry_replay import (BUNDLE, TRACE, load_and_verify,
                                    trace_word, verify_source)
except ImportError:
    from score import STRATA, rotate, necklace, select_manifest
    from telemetry_replay import BUNDLE, TRACE, load_and_verify, trace_word, verify_source


def reverse_bits(word: int, width: int) -> int:
    if width < 1 or not 0 <= word < (1 << width):
        raise ValueError("invalid binary history")
    return sum(((word >> bit) & 1) << (width - bit - 1) for bit in range(width))


def bracelet(word: int, width: int) -> int:
    """Canonical orbit representative under cyclic shifts and time reversal."""
    if width < 1 or not 0 <= word < (1 << width):
        raise ValueError("invalid binary history")
    return min(necklace(word, width), necklace(reverse_bits(word, width), width))


def descriptor(word: int, width: int, mode: str) -> int:
    if width < 1 or not 0 <= word < (1 << width):
        raise ValueError("history out of bounds")
    if mode == "ordered":
        return word
    if mode == "cyclic":
        return necklace(word, width)
    if mode == "dihedral":
        return bracelet(word, width)
    if mode == "count":
        return word.bit_count()
    raise ValueError("unknown descriptor")


def necklace_counts(width: int, weight: int) -> dict[str, int]:
    """Enumerate finite fixed-weight cyclic and dihedral orbit counts."""
    if not 1 <= width <= 16 or not 0 <= weight <= width:
        raise ValueError("outside exact finite enumeration budget")
    members = [w for w in range(1 << width) if w.bit_count() == weight]
    necklaces = {necklace(w, width) for w in members}
    bracelets = {bracelet(w, width) for w in members}
    fixed = {n for n in necklaces if necklace(reverse_bits(n, width), width) == n}
    assert len(necklaces) + len(fixed) == 2 * len(bracelets)
    return {"width": width, "weight": weight, "necklaces": len(necklaces),
            "reversal_fixed_necklaces": len(fixed), "bracelets": len(bracelets)}


def pair_and_decoder_ledger(words: list[int], labels: list[int], width: int) -> dict:
    """Exact complete evaluator restricted to unequal labels.

    Decoder values are *in-sample oracle upper bounds*: choose the majority
    label within each descriptor class, after seeing all these same labels.
    They are not held-out probe accuracy measurements.
    """
    if len(words) != len(labels) or not words or width < 1:
        raise ValueError("invalid group")
    modes = ("ordered", "cyclic", "dihedral", "count")
    maps = {mode: [descriptor(w, width, mode) for w in words] for mode in modes}
    result = {"eligible": 0}
    for mode in modes:
        class_counts = defaultdict(Counter)
        for index, label in enumerate(labels):
            class_counts[maps[mode][index]][label] += 1
        result[mode] = {
            "distinguished": 0,
            "classes": len(class_counts),
            "in_sample_oracle_correct": sum(max(counts.values()) for counts in class_counts.values()),
            "denominator": len(words),
        }
    for i, j in combinations(range(len(words)), 2):
        if labels[i] == labels[j]:
            continue
        result["eligible"] += 1
        for mode in modes:
            result[mode]["distinguished"] += maps[mode][i] != maps[mode][j]
    o, c, d, h = (result[k]["distinguished"] for k in modes)
    assert result["eligible"] >= o >= c >= d >= h
    assert all(result[a]["in_sample_oracle_correct"] >= result[b]["in_sample_oracle_correct"]
               for a, b in zip(modes, modes[1:]))
    result["phase_sensitive"] = o - c
    result["time_orientation_sensitive"] = c - d
    result["dihedral_beyond_count"] = d - h
    assert (o - h == result["phase_sensitive"] + result["time_orientation_sensitive"]
            + result["dihedral_beyond_count"])
    return result


def analyze(bundle: dict, trace: dict) -> dict:
    verify_source(bundle, trace)
    manifest = select_manifest(bundle)
    by_window = {}
    for width in (4, 8, 16):
        by_window[str(width)] = {}
        for group in STRATA:
            indices = [index for index, row in enumerate(bundle["items"])
                       if row["stratum"] == group]
            words = [trace_word(trace, index, width) for index in indices]
            labels = [int(bundle["items"][index]["target_digit"]) for index in indices]
            full = pair_and_decoder_ledger(words, labels, width)
            chosen_ids = {row["item_id"] for row in manifest["strata"][group]}
            selected = [index for index in indices if bundle["items"][index]["item_id"] in chosen_ids]
            small = pair_and_decoder_ledger(
                [trace_word(trace, index, width) for index in selected],
                [int(bundle["items"][index]["target_digit"]) for index in selected], width)
            by_window[str(width)][group] = {"all_64": full, "preregistered_nine": small}
    return {"schema": "COE-07R-pretrained-output-dihedral-gauge-v1",
            "status": "truth-conditioned native output oracle, post-hoc structural diagnostic",
            "model_checkpoint_sha256": trace["checkpoint_sha256"],
            "benchmark_sha256": bundle["bundle_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "orbit_capacity": {str(w): necklace_counts(8,w)
                               for w in (3,4,5)},
            "windows": by_window}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    # Validates the exact Git blob IDs of both committed source artifacts.
    load_and_verify()
    result = analyze(json.loads(BUNDLE.read_text()), json.loads(TRACE.read_text()))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "stalled": {T: result["windows"][T]["S4_stalled"]["all_64"]
                    for T in ("4", "8", "16")},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
