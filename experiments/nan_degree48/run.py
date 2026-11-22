"""Degree-conditioned readout wiring analysis of the archived GAUGE42 surrogate."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from importlib.metadata import version
from itertools import product
import json
import math
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_delay39.graph_host import graph_cells
from experiments.nan_gauge42.run import read_archive, extract_pair, seal as gauge_seal
from experiments.nan_gauge42.transport import fit, orthonormal_bank
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new

from .null import analyze, exact_distribution


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
READPATH = REPO / "experiments/nan_readpath37"
GAUGE = REPO / "experiments/nan_gauge42"
PRIOR_ARCHIVE = {
    "nf4": "0bd47195d3c292e2266611c9fd325762e17d03e4d375f5fad26f1ebc3cd6b93d",
    "bf16": "1593673dfd556e2bb3bcc519f24959ff6321af41318cf562d174a1a08537b239",
}


def registration():
    own = (".gitattributes", "__init__.py", "PROTOCOL.md", "registration.json", "run.py", "null.py")
    upstream = (
        "experiments/nan_gauge42/SOURCE_MANIFEST.json", "experiments/nan_gauge42/run.py",
        "experiments/nan_gauge42/transport.py", "experiments/nan_gauge42/registration.json",
        "experiments/nan_delay39/graph_host.py", "experiments/nan_readpath37/registration.json",
        "experiments/nan_readpath37/run.py", "experiments/nan_phase35/run.py",
        "experiments/nan_select47/select.py",
    )
    paths = [HERE / n for n in own] + [REPO / n for n in upstream]
    return {"experiment": "NAN-DEGREE-48", "GAUGE42_native_receipts_sha256": PRIOR_ARCHIVE,
            "files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in paths}}


def exhaustive_fixture(check):
    a, b = 2, 3
    families = defaultdict(list)
    for values in product((0, 1), repeat=a * b):
        rows = tuple(sum(values[i * b:(i + 1) * b]) for i in range(a))
        cols = tuple(sum(values[j::b]) for j in range(b))
        contacts = [(i, j) for i, j in product(range(a), range(b)) if values[i * b + j]]
        families[rows, cols].append(contacts)
    active_sets = [set(i for i in range(a) if bitset & (1 << i)) for bitset in range(2**a)]
    sink_sets = [set(j for j in range(b) if bitset & (1 << j)) for bitset in range(2**b)]
    tests = 0
    for (rows, cols), contacts in families.items():
        check()
        for src, sink in product(active_sets, sink_sets):
            expected = Counter()
            for graph in contacts:
                paths = [(i, j) for i, j in graph if i in src and j in sink]
                expected[len({i for i, _ in paths}), len({j for _, j in paths})] += 1
            actual = exact_distribution(rows, cols, src, sink)
            assert actual == expected
            tests += 1
    planted = exact_distribution([1, 1], [1, 1], {0}, {0})
    assert planted == {(0, 0): 1, (1, 1): 1}
    rows, cols = [1, 2, 3, 4], [1, 2, 3, 4]
    direct = exact_distribution(rows, cols, {0, 1}, {1, 2})
    reverse = exact_distribution(cols, rows, {1, 2}, {0, 1})
    assert direct == {(y, x): n for (x, y), n in reverse.items()}
    return {"enumerated_binary_graphs": 2**(a*b), "exact_conditioned_distribution_checks": tests,
            "balanced_matching_null_has_both_path_and_no_path": True,
            "transpose_symmetry": True}


def legacy_binding(old, direct, measured, cutoff):
    legacy = [c for c in old["graphs"]["native"] if c["occupancy_rank_fraction"] == cutoff]
    if len(legacy) != 1:
        raise ValueError("The archived native graph did not record this cutoff")
    saved = legacy[0]
    integer_keys = ("through_paths", "backbone_nodes", "local_in_out_nodes", "exact_membership_tie")
    real_keys = ("signed_path_product_sum", "absolute_path_product_sum")
    if any(saved[k] != direct[k] for k in integer_keys) or any(
            not math.isclose(saved[k], direct[k], abs_tol=1e-10, rel_tol=1e-9) for k in real_keys):
        raise ValueError("New archive reconstruction differs from the GAUGE42 native gauge receipt")
    if (sum(measured["observed_backbone"]) != saved["backbone_nodes"]
            or sum(measured["local_degree_positives"]) != saved["local_in_out_nodes"]):
        raise ValueError("Fixed degree analysis has a different observed graph")


def archived_backend(backend, base, cfg, gauge_cfg, roster, tokens, head, check, np):
    previous_bytes = bound(REPO / "runs/nan_gauge42/attempt01" / f"{backend}_private.json", PRIOR_ARCHIVE[backend])
    previous = {(r["split"], r["arm"], r["root"]): r for r in json.loads(previous_bytes)}
    twins, reads, native, max_logit = read_archive(backend, base, roster, head, check, np)
    root, inverse, whitened = orthonormal_bank(native, np)
    pairs = {key: extract_pair(key, worlds, tokens, head, reads, root, whitened, np)
             for key, worlds in sorted(twins.items())}
    calibration = [row for (split, _, _), row in pairs.items() if split == "calibration"]
    if len(calibration) != 8:
        raise ValueError("GAUGE42 CAL training roster changed")
    initial, final = (np.stack([pair[i]["coefficients"] for pair in calibration]) for i in (0, 1))
    transfer = fit(initial, final, gauge_cfg["ridge_factor"], np)
    writes = root @ whitened
    norms = np.linalg.norm(writes, axis=1)
    middle = ((root @ transfer @ inverse) * (norms[None, :] / norms[:, None])).tolist()
    results = []
    for key, pair in pairs.items():
        split, _, puzzle = key
        if split != "development":
            continue
        check()
        source = ((pair[0]["coefficients"] @ inverse) * norms).tolist()
        sink = ((writes @ pair[1]["gradient"]) / norms).tolist()
        direct = {g["occupancy_rank_fraction"]: g for g in graph_cells(source, middle, sink)}
        for cutoff in cfg["cutoffs"]:
            values = analyze(source, middle, sink, cutoff, check)
            legacy_binding(previous[key], direct[cutoff], values, cutoff)
            results.append({"root": puzzle, "cutoff": cutoff, **values})
    if len({x["root"] for x in results}) != 16 or len(results) != 16 * len(cfg["cutoffs"]):
        raise ValueError("Focused DEV graph roster does not close")
    return results, max_logit


def compare(rows):
    n = int(rows["null_configurations"])
    expectation = Fraction(int(rows["null_total_backbone_numerator"]), n)
    tail = Fraction(int(rows["null_upper_tail_count"]), n)
    spanning = Fraction(int(rows["null_spanning_count"]), n)
    return expectation, tail, spanning


def aggregate(rows, cutoffs):
    result = {}
    for c in cutoffs:
        use = [r for r in rows if r["cutoff"] == c]
        summaries = [compare(r) for r in use]
        effects = [sum(r["observed_backbone"]) - expected for r, (expected, _, _) in zip(use, summaries, strict=True)]
        result[str(c)] = {
            "registered_roots": len(use),
            "observed_spanning_roots": sum(any(r["observed_backbone"]) for r in use),
            "mean_conditional_spanning_probability": float(fmean(float(v[2]) for v in summaries)),
            "observed_equals_local_degree_roots": sum(r["observed_backbone"] == r["local_degree_positives"] for r in use),
            "mean_observed_backbone_nodes": fmean(sum(r["observed_backbone"]) for r in use),
            "mean_null_expected_backbone_nodes": fmean(float(e) for e, _, _ in summaries),
            "roots_observed_above_exact_null_mean": sum(v > 0 for v in effects),
            "roots_observed_equal_exact_null_mean": effects.count(0),
            "roots_observed_below_exact_null_mean": sum(v < 0 for v in effects),
            "median_conditional_tail_descriptive_only": median_tail([value[1] for value in summaries]),
            "tail_at_most_0_05_descriptive_count": sum(v[1] <= Fraction(1, 20) for v in summaries),
            "smallest_exact_null_support_size": str(min(int(r["null_configurations"]) for r in use)),
            "largest_exact_null_support_size": str(max(int(r["null_configurations"]) for r in use)),
        }
    return result


def median_tail(vals):
    order = sorted(vals)
    n = len(order)
    return float((order[n // 2] + order[(n - 1) // 2]) / 2)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        raise RuntimeError("Archive and finite proofs require verification assertions")
    if args.freeze:
        write_new(MANIFEST, registration())
        print(digest(MANIFEST))
        return
    if not (args.snapshot and args.output):
        cli.error("--snapshot and --output are required")
    if json.loads(MANIFEST.read_bytes()) != registration() or json.loads((GAUGE / "SOURCE_MANIFEST.json").read_bytes()) != gauge_seal():
        raise ValueError("Frozen graph-solver or GAUGE42 dependencies changed")
    base = json.loads((READPATH / "registration.json").read_bytes())
    cfg = json.loads((HERE / "registration.json").read_bytes())
    gauge_cfg = json.loads((GAUGE / "registration.json").read_bytes())
    if args.snapshot.name != base["revision"]:
        raise ValueError("Wrong frozen checkpoint")
    with lease(cfg) as (check, samples):
        if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("Pinned numerical packages changed")
        import numpy as np
        from tokenizers import Tokenizer

        math_check = exhaustive_fixture(check)
        bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        labels = {key: tok.encode(key, add_special_tokens=False).ids for key in "ABCDEFGHIJKLMNOP"}
        if any(len(ids) != 1 for ids in labels.values()) or len({ids[0] for ids in labels.values()}) != 16:
            raise ValueError("Bare label tokenization differs from GAUGE42")
        values, _ = joined(base)
        focus = {slot: row for slot, row in values.items() if row["arm"] == "focused"}
        details, totals = {}, {}
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            tokens = {key: ids[0] for key, ids in labels.items()}
            for backend in base["backends"]:
                details[backend], error = archived_backend(backend, base, cfg, gauge_cfg, focus, tokens, head, check, np)
                totals[backend] = {"cutoffs": aggregate(details[backend], cfg["cutoffs"]), "native_replay_max_logit_error": error}
                print(f"{backend}: verified {len(details[backend])} reconstructed fixed-degree records", flush=True)
        check()
        if "torch" in sys.modules:
            raise RuntimeError("Archive audit imported Torch")
    private = args.output / "private.json"
    write_new(private, details)
    write_new(args.output / "aggregate.json", {
        "experiment": cfg["experiment"], "status": "complete", "source_manifest_sha256": digest(MANIFEST),
        "private_audit_sha256": digest(private), "per_backend": totals, "integer_count_checks": math_check,
        "surrogate_calibration_roots": 8, "observable_DEV_roots_per_backend": 16,
        "new_pretrained_forwards": 0, "inferential_tests": 0, "sealed_test_accessed": False,
        "resource": {"max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
                     "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
                     "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples)},
    })


if __name__ == "__main__":
    main()
