"""Measure the finite rounded effects of vocabulary-aware BPD write directions."""
import argparse
from collections import defaultdict
from importlib.metadata import version
import json
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_dark44.head import full_head, verify_reference
from experiments.nan_dark44.run import read_winner, roster
from experiments.nan_delay39.instrument import prepare
from experiments.nan_gauge42.run import ARCHIVE, READPATH, read_archive
from experiments.nan_gauge42.transport import head_gradient
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new

from .spectral import dirs, projected_gram, verify


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
PRIOR = REPO / "experiments/nan_dark44"
DELAY = REPO / "experiments/nan_delay39"
SIGNS = {"positive": 1, "negative": -1}
ARMS = ("margin_null_donor", "least_sensitive_null", "regularized_null_donor")


def seal():
    paths = [HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "registration.json",
                                     "run.py", "spectral.py")]
    paths += [REPO / name for name in (
        "experiments/nan_dark44/head.py", "experiments/nan_dark44/run.py", "experiments/nan_dark44/registration.json",
        "experiments/nan_dark44/SOURCE_MANIFEST.json", "experiments/nan_gauge42/run.py",
        "experiments/nan_gauge42/transport.py", "experiments/nan_readpath37/registration.json",
        "experiments/nan_readpath37/run.py", "experiments/nan_delay39/instrument.py",
        "experiments/nan_delay39/registration.json", "experiments/nan_dose38/run.py",
        "experiments/nan_phase35/run.py", "experiments/nan_ram28/resources.py", "experiments/nan_ranksite13/resources.py")]
    return {"experiment": "NAN-SPECTRAL-45",
            "files_sha256": {path.relative_to(REPO).as_posix(): digest(path) for path in paths}}


def complementary_source(base, prior):
    old = {row["root"] for row in roster(base, prior).values()}
    records, _ = joined(base)
    out = {i: row for i, row in records.items()
           if row["split"] == "calibration" and row["arm"] == "focused" and row["root"] not in old}
    if len(out) != 8 or len({r["root"] for r in out.values()}) != 4 or len(old) != 4:
        raise ValueError("DARK44 and SPECTRAL45 source complements changed")
    return out


def archived_cases(base, source, dose, head, tokens, check, np):
    locations = {(row["root"], row["world"]): i for i, row in source.items()}
    rows, bases, components, replay = [], [], {}, {}
    dimensions = 0
    for backend, pin in base["backends"].items():
        twins, reads, writes, replay[backend] = read_archive(backend, base, source, head, check, np)
        idx = dose["phase1_components"]
        selected = writes[idx]
        basis, factor = np.linalg.qr(selected.T, mode="reduced")
        if any(abs(v) < 1e-10 for v in factor.diagonal()):
            raise ValueError("The pinned physical write span is rank deficient")
        bases.append(basis)
        components[backend] = (basis, list(range(dimensions, dimensions + basis.shape[1])))
        dimensions += basis.shape[1]
        audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", pin["audit"]))
        for (_, _, root), worlds in sorted(twins.items()):
            if set(worlds) != {"A", "B"}:
                raise ValueError("Independent root is missing its counterfactual twin")
            for world, other in (("A", "B"), ("B", "A")):
                check()
                answer, a, _ = worlds[world]
                foil, b, _ = worlds[other]
                h = a[1, "residual_after_layer"].astype(np.float64)
                local = a[1, "down_proj_input"].astype(np.float64)
                donor = b[1, "down_proj_input"].astype(np.float64)
                donor_write = ((donor - local) @ reads[:, idx]) @ selected
                pair = [tokens[foil], tokens[answer]]
                rows.append({"backend": backend, "root": root, "world": world, "tokens": pair,
                             "donor": donor_write, "h": h,
                             "archive": read_winner(backend, locations[root, world], audit)})
    return rows, components, np.column_stack([*bases, *(row["h"] for row in rows)]), dimensions, replay


def edits(rows, banks, gram, offset, head, cfg, dose, check, np):
    states, refs, labels, records = [], [], [], []
    for i, row in enumerate(rows):
        check()
        h, donor = row["h"], row["donor"]
        tokens = row["tokens"]
        basis, columns = banks[row["backend"]]
        gram_ids = [*columns, offset + i]
        grad_atoms = head.gamma.astype(np.float64) * (
            head.row(tokens[0]).astype(np.float64) - head.row(tokens[1]).astype(np.float64))
        grad = head_gradient(h, grad_atoms, head.eps, np)
        vectors, rms, eigen = dirs(gram[np.ix_(gram_ids, gram_ids)], basis, h, grad, donor,
                                   cfg["ridge_median_factor"], head.eps, np)
        alignment = {key: float(vector @ donor / np.linalg.norm(donor)) for key, vector in vectors.items()}
        baseline = len(states)
        states.append(h)
        refs.append(baseline)
        labels.append(tokens)
        result = {key: row[key] for key in ("backend", "root", "world", "archive")}
        result.update(clean_index=baseline, directional_margin_null_max=max(abs(float(w @ grad)) for w in vectors.values()),
                      eigenvalues=eigen.tolist(), continuous_rms_per_l2=rms, positive_donor_cosine=alignment, doses={})
        for fraction in cfg["doses"]:
            all_cases = {}
            for arm in ARMS:
                all_cases[arm] = {}
                for tag, sign in SIGNS.items():
                    patch = prepare(h, sign * vectors[arm], fraction, dose["dose_relative_tolerance"],
                                    dose["minimum_direction_cosine"], np)
                    item = {"eligible": patch.eligible, "index": None, "requested_l2": patch.requested_l2,
                            "reached_l2": patch.realized_l2, "direction_cosine": patch.cosine}
                    if patch.eligible:
                        item["index"] = len(states)
                        states.append(patch.after)
                        refs.append(baseline)
                        labels.append(tokens)
                    all_cases[arm][tag] = item
            result["doses"][str(fraction)] = all_cases
        records.append(result)
    return records, np.stack(states), np.asarray(refs), np.asarray(labels)


def evaluate(records, live, prior, dose):
    replay_top = replay_lz = 0.0
    for record in records:
        index = record.pop("clean_index")
        saved = record.pop("archive")
        if saved["free_top_token_id"] != live["winner"][index] or saved["vocabulary_size"] != live["vocab"]:
            raise ValueError("Unedited vocabulary winner differs from the pinned native archive")
        replay_top = max(replay_top, abs(saved["free_top_token_logit"] - float(live["top_logit"][index])))
        replay_lz = max(replay_lz, abs(saved["full_log_normalizer"] - float(live["logz"][index])))
        for options in record["doses"].values():
            control = options["margin_null_donor"]
            for directions in options.values():
                for sign, item in directions.items():
                    i = item.pop("index")
                    if i is None:
                        continue
                    contrast = abs(item["reached_l2"] - control[sign]["reached_l2"])
                    difference = live["bare"][i] - live["bare"][index]
                    margin = abs(float(difference[0] - difference[1]))
                    maximum = float(live["max_abs_delta"][i])
                    greedy = bool(live["winner"][i] == live["winner"][index])
                    item.update({
                        "matched_reference": control[sign]["eligible"] and contrast <=
                                             dose["max_contrast_dose_gap_fraction"] * item["requested_l2"],
                        "absolute_margin_delta": margin, "margin_pass": margin <= prior["task_margin_abs_tolerance"],
                        "full_pass": maximum <= prior["full_logit_abs_tolerance"] and greedy,
                        "greedy_unchanged": greedy, "max_abs_delta": maximum,
                        "rms_delta": float(live["rms_delta"][i]), "kl": float(live["KL_clean_to_edit"][i]),
                    })
    return {"max_abs_archived_top_logit_error": replay_top, "max_abs_archived_logz_error": replay_lz}


def average(values):
    return fmean(values) if values else None


def summarize(rows, cfg):
    roots = defaultdict(list)
    for row in rows:
        roots[row["root"]].append(row)
    if len(roots) != 4 or any(len(group) != 2 or {row["world"] for row in group} != {"A", "B"}
                              for group in roots.values()):
        raise ValueError("Incomplete independent root coverage")
    alignment = cfg["minimum_donor_cosine"]
    result = {"roots": len(roots), "native_worlds": len(rows),
              "minimum_gram_eigenvalue": min(row["eigenvalues"][0] for row in rows),
              "maximum_gram_eigenvalue": max(row["eigenvalues"][-1] for row in rows),
              "max_continuous_margin_derivative": max(row["directional_margin_null_max"] for row in rows),
              "directions": {}, "doses": {}}
    for arm in ARMS:
        values = [row["positive_donor_cosine"][arm] for row in rows]
        result["directions"][arm] = {
            "mean_cosine_to_donor": fmean(values),
            "mean_continuous_rms_per_unit_l2": fmean(row["continuous_rms_per_l2"][arm] for row in rows),
            "donor_aligned_worlds": sum(value >= alignment for value in values),
        }
    for number in cfg["doses"]:
        fraction = str(number)
        cond = result["doses"][fraction] = {}
        for arm in ARMS:
            observed = [(row, item) for row in rows for item in row["doses"][fraction][arm].values()
                        if item["eligible"]]
            matched = [(row, s, item) for row in rows for s, item in row["doses"][fraction][arm].items()
                       if item.get("matched_reference")]
            qualified_roots = sum(all(item.get("full_pass", False) and item.get("margin_pass", False)
                for row in group for item in row["doses"][fraction][arm].values()) for group in roots.values())
            cond[arm] = {
                "registered_signed_edits": 2 * len(rows), "physically_eligible": len(observed),
                "matched_reference": len(matched),
                "margin_preserved": sum(x["margin_pass"] for _, x in observed),
                "full_head_preserved": sum(x["full_pass"] for _, x in observed),
                "full_head_and_margin_preserved": sum(x["full_pass"] and x["margin_pass"] for _, x in observed),
                "all_worlds_signs_both_gates_roots": qualified_roots,
                "full_margin_and_donor_aligned": sum(x["full_pass"] and x["margin_pass"] and
                                                    row["positive_donor_cosine"][arm] >= alignment for row, x in observed),
                "mean_rms_delta": average([x["rms_delta"] for _, x in observed]),
                "mean_max_abs_delta": average([x["max_abs_delta"] for _, x in observed]),
                "mean_kl": average([x["kl"] for _, x in observed]),
                "mean_abs_margin_delta": average([x["absolute_margin_delta"] for _, x in observed]),
                "matched_has_smaller_rms": sum(x["rms_delta"] <
                    row["doses"][fraction]["margin_null_donor"][s]["rms_delta"] for row, s, x in matched),
                "matched_has_smaller_peak": sum(x["max_abs_delta"] <
                    row["doses"][fraction]["margin_null_donor"][s]["max_abs_delta"] for row, s, x in matched),
            }
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        raise RuntimeError("The checkpoint head loader requires assertions")
    if args.freeze:
        write_new(MANIFEST, seal())
        print(digest(MANIFEST))
        return
    if not (args.snapshot and args.output):
        cli.error("--snapshot and --output are required")
    if json.loads(MANIFEST.read_bytes()) != seal():
        raise ValueError("SPECTRAL45 runtime manifest changed")
    base = json.loads((READPATH / "registration.json").read_bytes())
    prior = json.loads((PRIOR / "registration.json").read_bytes())
    cfg = json.loads((HERE / "registration.json").read_bytes())
    dose = json.loads((DELAY / "registration.json").read_bytes())
    if args.snapshot.name != base["revision"] or args.snapshot.name != dose["checkpoint_revision"]:
        raise ValueError("Snapshot revision mismatch")
    with lease(cfg) as (check, samples):
        if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("Numerical or tokenizer runtime pin changed")
        import numpy as np
        from tokenizers import Tokenizer

        analytic_error = verify(np)
        fixture_rows = verify_reference(np)
        bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        labels = {key: tokenizer.encode(key, add_special_tokens=False).ids for key in "ABCDEFGHIJKLMNOP"}
        if any(len(ids) != 1 for ids in labels.values()) or len({ids[0] for ids in labels.values()}) != 16:
            raise ValueError("Checkpoint bare label tokens changed")
        source = complementary_source(base, prior)
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            tokens = {label: ids[0] for label, ids in labels.items()}
            rows, banks, vectors, offset, replay = archived_cases(base, source, dose, head, tokens, check, np)
            gram = projected_gram(head, vectors, prior["block_rows"], check, np)
            cases, states, refs, row_labels = edits(rows, banks, gram, offset, head, cfg, dose, check, np)
            stats = full_head(head, states, refs, row_labels, prior["block_rows"], check, np)
            closure = evaluate(cases, stats, prior, dose)
            if max(closure.values()) > base["native_logit_abs_tol"]:
                raise ValueError("Clean native head replay exceeded tolerance")
        check()
        if "torch" in sys.modules:
            raise RuntimeError("The spectral CPU audit imported Torch")
    output = args.output / "private.json"
    write_new(output, cases)
    write_new(args.output / "aggregate.json", {
        "experiment": cfg["experiment"], "status": "complete", "source_manifest_sha256": digest(MANIFEST),
        "private_receipt_sha256": digest(output), "DARK44_CAL_roots_reused": 0,
        "vocabulary_rows": stats["vocab"], "qualified_residual_rows_including_clean": len(states),
        "vocabulary_Gram_columns": gram.shape[1], "fixture_max_derivative_difference": analytic_error,
        "fixture_streaming_reduction_rows": fixture_rows, "native_selected_replay_error_by_backend": replay, **closure,
        "per_backend": {b: summarize([row for row in cases if row["backend"] == b], cfg) for b in base["backends"]},
        "new_pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "resource": {"max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
                     "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
                     "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples)},
    })
    print(f"Verified rank-limited vocabulary Gram and {len(states)} finite full-head rows", flush=True)


if __name__ == "__main__":
    main()
