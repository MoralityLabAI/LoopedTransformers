"""Test finite margin preservation against the entire pinned Nanbeige head."""
import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_delay39.instrument import prepare
from experiments.nan_gauge42.run import ARCHIVE, READPATH, read_archive
from experiments.nan_gauge42.transport import head_gradient
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new

from .head import full_head, verify_reference


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
DELAY = REPO / "experiments/nan_delay39"
ARMS = ("twin_donor", "margin_null_donor", "visible_span")
SIGNS = {"positive": 1, "negative": -1}


def seal():
    paths = [HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "registration.json",
                                     "run.py", "head.py")]
    paths += [REPO / name for name in (
        "experiments/nan_gauge42/run.py", "experiments/nan_gauge42/transport.py",
        "experiments/nan_readpath37/run.py", "experiments/nan_readpath37/registration.json",
        "experiments/nan_readpath37/SOURCE_MANIFEST.json", "experiments/nan_delay39/registration.json",
        "experiments/nan_delay39/instrument.py", "experiments/nan_dose38/run.py",
        "experiments/nan_phase35/run.py", "experiments/nan_ram28/resources.py",
        "experiments/nan_ranksite13/resources.py")]
    return {"experiment": "NAN-DARK-44",
            "files_sha256": {path.relative_to(REPO).as_posix(): digest(path) for path in paths}}


def roster(base, cfg):
    all_rows, _ = joined(base)
    focused = {i: row for i, row in all_rows.items()
               if row["split"] == "calibration" and row["arm"] == "focused"}
    roots = {row["root"] for row in focused.values()}
    if len(roots) != base["roots"]["calibration"]:
        raise ValueError("The registered CAL source pool changed")
    order = lambda root: sha256(f"{cfg['roster_seed']}|{root}".encode()).digest()
    chosen = set(sorted(roots, key=order)[:cfg["calibration_roots"]])
    use = {i: row for i, row in focused.items() if row["root"] in chosen}
    if len(use) != 2 * cfg["calibration_roots"]:
        raise ValueError("CAL twin roster is incomplete")
    return use


def span_directions(donor, gradient, writes, np):
    basis, factor = np.linalg.qr(writes.T, mode="reduced")
    if any(abs(value) < 1e-10 for value in factor.diagonal()):
        raise ValueError("Pinned write span lost physical rank")
    common = basis @ (basis.T @ gradient)
    strength = float(np.linalg.norm(common))
    donor_norm = float(np.linalg.norm(donor))
    if strength < 1e-12 or donor_norm < 1e-12:
        return None, None, None
    visible = common / strength
    dark = donor - float(donor @ visible) * visible
    if np.linalg.norm(dark) < 1e-12:
        dark = np.zeros_like(donor)
    leakage = float(dark @ gradient)
    if abs(leakage) > 1e-10:
        raise ValueError("Continuous margin-null direction does not close")
    return dict(zip(ARMS, (donor, dark, visible))), float(np.linalg.norm(dark) / donor_norm), abs(leakage)


def read_winner(backend, slot, audit):
    case = json.loads(bound(ARCHIVE / backend / f"case_{slot:04d}.json", audit["records"][slot]["case_sha256"]))
    phase = case["initial_phases"][0]
    if phase["loop"] != 1:
        raise ValueError("Winner is for the wrong physical pass")
    return {key: phase["score"][key] for key in ("free_top_token_id", "free_top_token_logit",
                                                 "full_log_normalizer", "vocabulary_size")}


def prepare_rows(backend, base, source, dose, tokens, head, check, np):
    pairs, reads, writes, replay = read_archive(backend, base, source, head, check, np)
    ids = {(row["root"], row["world"]): slot for slot, row in source.items()}
    indices = dose["phase1_components"]
    selected = writes[indices]
    pinned = base["backends"][backend]
    audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", pinned["audit"]))
    rows = []
    states, labels, references = [], [], []

    def add(state, two_tokens, ref=None):
        index = len(states)
        states.append(state)
        labels.append(two_tokens)
        references.append(index if ref is None else ref)
        return index

    for (_, _, root), worlds in sorted(pairs.items()):
        if set(worlds) != {"A", "B"}:
            raise ValueError("Incomplete CAL twin pair")
        for world, other in (("A", "B"), ("B", "A")):
            check()
            answer, states_a, _ = worlds[world]
            donor_answer, states_b, _ = worlds[other]
            h = states_a[1, "residual_after_layer"].astype(np.float64)
            a, b = (item[1, "down_proj_input"].astype(np.float64) for item in (states_a, states_b))
            tokens_pair = [tokens[donor_answer], tokens[answer]]
            head_dir = head.gamma.astype(np.float64) * (
                head.row(tokens_pair[0]).astype(np.float64) - head.row(tokens_pair[1]).astype(np.float64))
            gradient = head_gradient(h, head_dir, head.eps, np)
            donor = ((b - a) @ reads[:, indices]) @ selected
            vectors, retained, leak = span_directions(donor, gradient, selected, np)
            original = add(h, tokens_pair)
            outcome = {
                "backend": backend, "root": root, "world": world, "clean_index": original,
                "archive": read_winner(backend, ids[root, world], audit),
                "retained_donor_norm_fraction": retained, "continuous_margin_null_abs_residual": leak,
                "arms": {},
            }
            for arm in ARMS:
                outcome["arms"][arm] = {}
                for sign, scale in SIGNS.items():
                    record = {"qualified": False, "index": None}
                    if vectors is not None:
                        patch = prepare(h, scale * vectors[arm], dose["phase1_dose_fraction"],
                                        dose["dose_relative_tolerance"], dose["minimum_direction_cosine"], np)
                        record.update({"qualified": patch.eligible, "requested_l2": patch.requested_l2,
                                       "reached_l2": patch.realized_l2, "direction_cosine": patch.cosine})
                        if patch.eligible:
                            record["index"] = add(patch.after, tokens_pair, original)
                    outcome["arms"][arm][sign] = record
            rows.append(outcome)
    return rows, states, labels, references, replay


def score(rows, measured, base, cfg, dose, np):
    native_logit_error, native_logz_error = 0.0, 0.0
    for row in rows:
        i = row.pop("clean_index")
        observed = row.pop("archive")
        native_logit_error = max(native_logit_error, abs(float(measured["top_logit"][i]) - observed["free_top_token_logit"]))
        native_logz_error = max(native_logz_error, abs(float(measured["logz"][i]) - observed["full_log_normalizer"]))
        if int(measured["winner"][i]) != observed["free_top_token_id"] or measured["vocab"] != observed["vocabulary_size"]:
            raise ValueError("Clean full-head winner or vocabulary changed")
        clean = measured["bare"][i]
        for arm in ARMS:
            for sign in SIGNS:
                item = row["arms"][arm][sign]
                j = item.pop("index")
                if j is None:
                    continue
                change = float((measured["bare"][j] - clean) @ np.array([1., -1.]))
                maximum = float(measured["max_abs_delta"][j])
                greedy = bool(measured["winner"][j] == measured["winner"][i])
                item.update({
                    "margin_delta": change, "margin_pass": abs(change) <= cfg["task_margin_abs_tolerance"],
                    "full_pass": maximum <= cfg["full_logit_abs_tolerance"]
                                 and (greedy or not cfg["winner_required_for_full_gate"]),
                    "greedy_same": greedy, "full_max_abs_delta": maximum, "full_rms_delta": float(measured["rms_delta"][j]),
                    "full_KL_clean_to_edit": float(measured["KL_clean_to_edit"][j]),
                })
        dark = row["arms"]["margin_null_donor"]
        visible = row["arms"]["visible_span"]
        row["matched_dark_visible"] = {sign: (dark[sign]["qualified"] and visible[sign]["qualified"] and
            abs(dark[sign]["reached_l2"] - visible[sign]["reached_l2"])
            <= dose["max_contrast_dose_gap_fraction"] * dark[sign]["requested_l2"]) for sign in SIGNS}
    if max(native_logit_error, native_logz_error) > base["native_logit_abs_tol"]:
        raise ValueError("Clean checkpoint head did not meet archive replay tolerance")
    return native_logit_error, native_logz_error


def summaries(data):
    report = {}
    groups = defaultdict(list)
    for row in data:
        groups[row["root"]].append(row)
    if any({row["world"] for row in g} != {"A", "B"} or len(g) != 2 for g in groups.values()):
        raise ValueError("Expected both CAL worlds for all roots")
    for arm in ARMS:
        items = [item for row in data for item in row["arms"][arm].values()]
        verified = [item for item in items if item["qualified"]]
        report[arm] = {
            "registered_signed_edits": len(items), "physically_qualified_signed_edits": len(verified),
            "margin_preserved": sum(item["margin_pass"] for item in verified),
            "full_head_preserved": sum(item["full_pass"] for item in verified),
            "unrestricted_winner_preserved": sum(item["greedy_same"] for item in verified),
            "margin_all_worlds_signs_roots": sum(all(item.get("margin_pass", False)
                for row in group for item in row["arms"][arm].values()) for group in groups.values()),
            "full_all_worlds_signs_roots": sum(all(item.get("full_pass", False)
                for row in group for item in row["arms"][arm].values()) for group in groups.values()),
            "mean_abs_margin_change": fmean(abs(item["margin_delta"]) for item in verified) if verified else None,
        } | {f"mean_{key}": fmean(item[key] for item in verified) if verified else None
             for key in ("full_max_abs_delta", "full_rms_delta", "full_KL_clean_to_edit")}
    pairs = [(row["arms"]["margin_null_donor"][s], row["arms"]["visible_span"][s])
             for row in data for s in SIGNS if row["matched_dark_visible"][s]]
    retained = [row["retained_donor_norm_fraction"] for row in data if row["retained_donor_norm_fraction"] is not None]
    leakage = [row["continuous_margin_null_abs_residual"] for row in data
               if row["continuous_margin_null_abs_residual"] is not None]
    return {
        "roots": len(groups), "worlds": len(data), "arms": report,
        "matched_dark_visible_signed_edits": len(pairs),
        "matched_dark_has_smaller_absolute_margin": sum(abs(a["margin_delta"]) < abs(b["margin_delta"]) for a, b in pairs),
        "matched_dark_has_smaller_full_rms": sum(a["full_rms_delta"] < b["full_rms_delta"] for a, b in pairs),
        "mean_retained_donor_norm_fraction": fmean(retained) if retained else None,
        "max_continuous_margin_null_abs_residual": max(leakage) if leakage else None,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        raise RuntimeError("The checkpoint-head loader checks require assertions")
    if args.freeze:
        write_new(MANIFEST, seal())
        print(digest(MANIFEST))
        return
    if not (args.snapshot and args.output):
        cli.error("--snapshot and --output are required")
    if json.loads(MANIFEST.read_bytes()) != seal():
        raise ValueError("DARK44 runtime manifest changed")
    base = json.loads((READPATH / "registration.json").read_bytes())
    cfg = json.loads((HERE / "registration.json").read_bytes())
    dose = json.loads((DELAY / "registration.json").read_bytes())
    if args.snapshot.name != base["revision"] or args.snapshot.name != dose["checkpoint_revision"]:
        raise ValueError("Snapshot revision mismatch")
    with lease(cfg) as (check, samples):
        if (version("numpy"), version("tokenizers")) != ("2.5.5", "0.20.3"):
            raise ValueError("Pinned numerical or decoder version changed")
        import numpy as np
        from tokenizers import Tokenizer

        fixture_rows = verify_reference(np)
        bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tokenizer = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        label_ids = {label: tokenizer.encode(label, add_special_tokens=False).ids for label in "ABCDEFGHIJKLMNOP"}
        if any(len(v) != 1 for v in label_ids.values()) or len({v[0] for v in label_ids.values()}) != 16:
            raise ValueError("Bare-label token IDs changed")
        source = roster(base, cfg)
        args.output.mkdir(parents=True, exist_ok=False)
        data, residue, labels, references, replay = [], [], [], [], {}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            tokens = {label: ids[0] for label, ids in label_ids.items()}
            for backend in base["backends"]:
                rows, states, selected, refs, replay[backend] = prepare_rows(backend, base, source, dose, tokens,
                                                                            head, check, np)
                offset = len(residue)
                for row in rows:
                    row["clean_index"] += offset
                    for arm in row["arms"].values():
                        for item in arm.values():
                            if item["index"] is not None:
                                item["index"] += offset
                data.extend(rows)
                residue.extend(states)
                labels.extend(selected)
                references.extend(offset + i for i in refs)
            stats = full_head(head, np.stack(residue), np.array(references), np.array(labels), cfg["block_rows"], check, np)
            logit_error, logz_error = score(data, stats, base, cfg, dose, np)
        write_new(args.output / "private.json", data)
        check()
        if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
            raise RuntimeError("The full head CPU check unexpectedly imported Torch")
    write_new(args.output / "aggregate.json", {
        "experiment": cfg["experiment"], "status": "complete", "source_manifest_sha256": digest(MANIFEST),
        "private_outcomes_sha256": digest(args.output / "private.json"),
        "vocabulary_rows": stats["vocab"], "qualified_residual_rows_evaluated_including_clean": len(residue),
        "clean_native_max_top_logit_error": logit_error, "clean_native_max_full_logZ_error": logz_error,
        "native_selected_logit_replay_max_by_backend": replay, "dense_comparison_fixture_rows": fixture_rows,
        "new_pretrained_forwards": 0, "formal_comparisons": 0, "sealed_test_accessed": False,
        "per_backend": {tag: summaries([row for row in data if row["backend"] == tag]) for tag in base["backends"]},
        "resource": {"max_sampled_private_bytes": max(item["private_bytes"] for item in samples),
                     "max_sampled_working_set_bytes": max(item["working_set_bytes"] for item in samples),
                     "min_sampled_available_RAM_bytes": min(item["available_physical_bytes"] for item in samples),
                     "sample_count": len(samples)},
    })
    print(f"Verified {len(residue)} full-head rows across {stats['vocab']} vocabulary entries", flush=True)


if __name__ == "__main__":
    main()
