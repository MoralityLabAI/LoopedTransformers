"""Replay pinned COMP14 endpoint states, without model or GPU calls."""
import argparse
from collections import defaultdict
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
from statistics import fmean, median
import sys

from experiments.nan_gauge42.run import ARCHIVE, READPATH
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bf16, bound, digest, joined, write_new
from experiments.nan_refit49.run import FIT_REPORT, FIT, load_fits

from .algebra import account, basis, carry_radius, continuous, margins, self_test

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
SITES = ("residual_after_layer", "down_proj_input", "down_proj_output")
COMPONENTS = ("in_span", "off_span", "effective_carry", "rms_scale", "bf16_remainder")


def seal():
    paths = [HERE / name for name in (".gitattributes", "__init__.py", "algebra.py",
                                     "run.py", "PROTOCOL.md", "registration.json")]
    paths += [READPATH / "registration.json", READPATH / "SOURCE_MANIFEST.json", FIT_REPORT,
              FIT / "SOURCE_MANIFEST.json"]
    paths += [REPO / name for name in (
        "experiments/nan_phase35/run.py", "experiments/nan_readpath37/run.py",
        "experiments/nan_refit49/run.py", "experiments/nan_gauge42/run.py",
        "experiments/nan_lattice53/run.py", "experiments/nan_ram28/resources.py",
        "experiments/nan_ranksite13/resources.py")]
    return {"experiment": "NAN-CARRY-63",
            "files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in paths}}


def read_archive(backend, base, roster, head, check, np):
    pin = base["backends"][backend]
    folder = ARCHIVE / backend
    audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", pin["audit"]))
    manifest = json.loads(bound(folder / "manifest.json", pin["manifest"]))
    if ((audit["status"], audit["worker_status"], audit["missing_contexts"], audit["contexts_checked"])
            != ("passed", "completed", 0, 240) or len(audit["records"]) != 240
            or manifest["checkpoint_revision"] != base["revision"]
            or manifest["target_parameter_sha256"] != pin["target"]):
        raise ValueError("COMP14 archive lacks its expected native audit")
    twins = defaultdict(dict)
    logit_error = 0.0
    for slot, row in roster.items():
        check()
        source = audit["records"][slot]
        case = json.loads(bound(folder / f"case_{slot:04d}.json", source["case_sha256"]))
        if any(row[k] != source[k] or row[k] != case[k] for k in ("id", "root", "world", "split", "arm")):
            raise ValueError("Captured case differs from its source identity")
        name = f"case_{slot:04d}_states.npz"
        if (case["state_file"], case["state_file_sha256"]) != (name, source["state_sha256"]):
            raise ValueError("Captured state pack differs from archive custody")
        entries = [i for i in case["state_index"] if i["layer_zero_based"] == 21 and i["site"] in SITES]
        keys = {(i["loop"], i["site"]): i["array_key"] for i in entries}
        if len(entries) != 6 or set(keys) != {(phase, site) for phase in (1, 2) for site in SITES}:
            raise ValueError("Layer-21 state capture has wrong sites or passes")
        states = {}
        with np.load(BytesIO(bound(folder / name, source["state_sha256"])), allow_pickle=False) as packed:
            for key, array_key in keys.items():
                value = packed[array_key]
                width = base["input_width"] if key[1] == "down_proj_input" else base["width"]
                if (value.dtype != np.float32 or value.shape != (width,) or not np.isfinite(value).all()
                        or not np.array_equal(bf16(value, np), value)):
                    raise ValueError(f"Expected finite BF16 capture at {key[1]}")
                states[key] = value.astype(np.float64)

        if [capture["loop"] for capture in case["initial_phases"]] != [1, 2]:
            raise ValueError("Missing archived native scores for both phases")
        for capture in case["initial_phases"]:
            score = capture["score"]
            stored = {int(key): value for key, value in score["selected_logits"].items()}
            stored[score["free_top_token_id"]] = score["free_top_token_logit"]
            replay = head.logits(states[capture["loop"], SITES[0]], list(stored))
            logit_error = max(logit_error, float(np.max(abs(replay - list(stored.values())))))
        group = twins[row["split"], row["arm"], row["root"]]
        if row["world"] in group:
            raise ValueError("Repeated world in twin source")
        group[row["world"]] = (row["answer"], case["initial_prompt_ids_sha256"], states)
    if logit_error > base["native_logit_abs_tol"] or len(twins) * 2 != len(roster):
        raise ValueError("Native replay or registered twin coverage differs")
    return twins, logit_error


def phase_metrics(phase, a, b, direction, tokens, head, projectors, factors, gauge, np):
    h = np.stack([s[phase, SITES[0]] for s in (a, b)])
    delta_x = a[phase, SITES[1]] - b[phase, SITES[1]]
    delta_y = a[phase, SITES[2]] - b[phase, SITES[2]]
    native_a, native_b = margins(head, h, tokens, np)
    native = float(native_a - native_b)
    scaled, rms, smooth = continuous(h, direction, head.eps, np)
    remainder = native - smooth
    bound_error = float((carry_radius(h[0], np) + carry_radius(h[1], np)) @ abs(scaled))
    condition = {}
    for name, q in projectors.items():
        parts, finite, measured, coalition = account(head, h, delta_y, q, tokens, scaled, native, np)
        parts.update(rms_scale=rms, bf16_remainder=remainder)
        closure = abs(sum(parts.values()) - native)
        if closure > 1e-10:
            raise ArithmeticError("Continuous endpoint and finite remainder do not close")
        result = {"continuous": parts, "finite": finite, "closure": closure,
                  "in_span_energy": float(measured @ measured)}
        if name in factors:
            v, u = factors[name]
            predicted = (delta_x @ v) @ u
            result["predicted_numerator"] = float(predicted @ scaled)
            result["predicted_minus_native_write_energy"] = float(np.linalg.norm(predicted - delta_y)**2)
        if name == "fitted":
            _, _, rotated, rotated_coalition = account(head, h, delta_y, gauge, tokens, scaled, native, np)
            result["gauge_projection_error"] = float(np.max(abs(measured - rotated)))
            result["gauge_coalition_error"] = float(np.max(abs(coalition - rotated_coalition)))
            if result["gauge_projection_error"] > 1e-10:
                raise ArithmeticError("Rank-eight physical projection changed after gauge mixing")
        condition[name] = result
    return {"native": native, "write_numerator": float(delta_y @ scaled),
            "write_energy": float(delta_y @ delta_y), "carry_numerator_bound": bound_error,
            "conditions": condition}


def growth(initial, final):
    return {name: final[name] - initial[name] for name in initial}


def summarize(pairs, random_names):
    phase1, phase2 = zip(*pairs, strict=True)
    native = [b["native"] - a["native"] for a, b in pairs]
    denom = sum(map(abs, native))
    native_y = [b["write_numerator"] - a["write_numerator"] for a, b in pairs]

    def average(values):
        return fmean(abs(v) for v in values)

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    groups = {}
    projection_gains = {}
    for name in phase1[0]["conditions"]:
        part = [(a["conditions"][name], b["conditions"][name]) for a, b in pairs]
        continuous_gain = [growth(a["continuous"], b["continuous"]) for a, b in part]
        finite_gain = [growth(a["finite"], b["finite"]) for a, b in part]
        projection_gains[name] = [abs(row["in_span"]) for row in finite_gain]
        item = {
            "mean_signed_continuous_gain": {k: fmean(v[k] for v in continuous_gain) for k in COMPONENTS},
            "mean_absolute_continuous_gain": {k: average(v[k] for v in continuous_gain) for k in COMPONENTS},
            "absolute_continuous_gain_over_native": {
                k: ratio(sum(abs(v[k]) for v in continuous_gain), denom) for k in COMPONENTS},
            "mean_signed_finite_gain": {k: fmean(v[k] for v in finite_gain) for k in finite_gain[0]},
            "mean_absolute_finite_gain": {k: average(v[k] for v in finite_gain) for k in finite_gain[0]},
            "absolute_finite_gain_over_native": {
                k: ratio(sum(abs(v[k]) for v in finite_gain), denom) for k in finite_gain[0]},
            "in_span_gain_same_native_sign_roots": sum(n * v["in_span"] > 0 for n, v in zip(native, finite_gain)),
            "pass2_write_energy_projected_fraction": ratio(
                sum(b["in_span_energy"] for _, b in part), sum(b["write_energy"] for b in phase2)),
        }
        if name in ("fitted", "initializer"):
            predicted = [b["predicted_numerator"] - a["predicted_numerator"] for a, b in part]
            item["predictor"] = {
                "pass2_relative_write_squared_error": ratio(
                    sum(b["predicted_minus_native_write_energy"] for _, b in part),
                    sum(b["write_energy"] for b in phase2)),
                "gain_numerator_relative_squared_error_vs_zero": ratio(
                    sum((p - y)**2 for p, y in zip(predicted, native_y)), sum(y**2 for y in native_y)),
                "mean_absolute_predicted_numerator_gain": average(predicted),
            }
        groups[name] = item

    carry = [b["conditions"]["fitted"]["continuous"]["effective_carry"] -
             a["conditions"]["fitted"]["continuous"]["effective_carry"] for a, b in pairs]
    bounds = [a["carry_numerator_bound"] + b["carry_numerator_bound"] for a, b in pairs]
    return {
        "roots": len(pairs), "mean_signed_native_gain": fmean(native), "mean_absolute_native_gain": average(native),
        "mean_absolute_pass1_native": average(a["native"] for a in phase1),
        "mean_absolute_pass2_native": average(b["native"] for b in phase2),
        "mean_absolute_native_write_numerator_gain": average(native_y),
        "mean_carry_numerator_gain_rounding_bound": fmean(bounds),
        "max_carry_numerator_gain_rounding_bound": max(bounds),
        "carry_numerator_gain_exceeds_rounding_bound_roots": sum(abs(c) > e for c, e in zip(carry, bounds)),
        "fitted_larger_than_initializer_finite_gain_roots": sum(
            f > s for f, s in zip(projection_gains["fitted"], projection_gains["initializer"])),
        "fitted_larger_than_random_median_finite_gain_roots": sum(
            f > median(projection_gains[name][i] for name in random_names)
            for i, f in enumerate(projection_gains["fitted"])),
        "conditions": groups,
    }


def evaluate(backend, base, config, head, roster, tokens, banks, output, check, np):
    twins, native_error = read_archive(backend, base, roster, head, check, np)
    initializer, fitted = banks[f"{backend}/T2"]
    factors = {"initializer": initializer, "fitted": fitted}
    projectors = {name: basis(u, np) for name, (_, u) in factors.items()}
    seeds = config["random_seeds"]
    random_names = [f"random_{seed}" for seed in seeds]
    for name, seed in zip(random_names, seeds, strict=True):
        projectors[name] = basis(np.random.default_rng(seed).normal(size=fitted[1].shape), np)
    rotation = basis(np.random.default_rng(config["gauge_seed"]).normal(size=(base["rank"], base["rank"])), np)
    gauge = basis((rotation * np.geomspace(.25, 4, base["rank"])[:, None]) @ fitted[1], np)

    groups = defaultdict(list)
    closure = defaultdict(float)
    private = []
    for (split, arm, root), worlds in sorted(twins.items()):
        check()
        if set(worlds) != {"A", "B"}:
            raise ValueError("Registered root lacks one counterfactual world")
        label_a, hash_a, a = worlds["A"]
        label_b, hash_b, b = worlds["B"]
        assert label_a != label_b
        if arm == "focused_erased" and (hash_a != hash_b or any(not np.array_equal(a[k], b[k]) for k in a)):
            raise ValueError("Erased twin states should replay identically")
        ids = [tokens[label_a], tokens[label_b]]
        direction = head.gamma.astype(np.float64) * (head.row(ids[0]).astype(np.float64) - head.row(ids[1]))
        pair = [phase_metrics(p, a, b, direction, ids, head, projectors, factors, gauge, np) for p in (1, 2)]
        if arm == "focused_erased" and any(p["native"] or p["write_numerator"] for p in pair):
            raise ArithmeticError("Erased twin endpoint contrast differs from zero")
        for p in pair:
            for c in p["conditions"].values():
                for key in ("closure", "gauge_projection_error", "gauge_coalition_error"):
                    closure[key] = max(closure[key], c.get(key, 0.0))
        groups[split, arm].append(pair)
        private.append({"root": root, "split": split, "arm": arm, "phases": pair})
    assert {k: len(v) for k, v in groups.items()} == {
        (split, arm): n for split, n in base["roots"].items() for arm in base["arms"]}
    write_new(output / f"{backend}_private.json", private)
    return {"replayed_states": len(roster) * 2, "native_replay_max_abs_error": native_error,
            "max_errors": dict(closure),
            "groups": {"/".join(k): summarize(pairs, random_names) for k, pairs in groups.items()}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        cli.error("Verification requires Python assertions")
    if args.freeze:
        write_new(MANIFEST, seal())
        return
    if not args.snapshot or not args.output:
        cli.error("--snapshot and --output are required for archive replay")
    if seal() != json.loads(MANIFEST.read_bytes()):
        raise ValueError("Frozen source differs from its manifest")
    cfg = json.loads((HERE / "registration.json").read_bytes())
    base = json.loads((READPATH / "registration.json").read_bytes())
    if args.snapshot.name != cfg["revision"] or base["revision"] != cfg["revision"]:
        cli.error("Archive and snapshot revision differ")
    with lease(cfg) as (check, samples):
        assert (version("numpy"), version("tokenizers")) == ("2.5.5", "0.20.3")
        import numpy as np
        from tokenizers import Tokenizer

        self_test(np)
        config = json.loads(bound(args.snapshot / "config.json", base["snapshot_pins"]["config.json"]))
        definitions = bound(args.snapshot / "configuration_nanbeige.py", cfg["configuration_sha256"])
        assert b"enable_hyper_connection=False," in definitions
        assert not config.get("enable_hyper_connection", False) and config["torch_dtype"] == "bfloat16"
        bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        codes = {label: tok.encode(label, add_special_tokens=False).ids for label in "ABCDEFGHIJKLMNOP"}
        assert all(len(code) == 1 for code in codes.values()) and len({v[0] for v in codes.values()}) == 16
        tokens = {label: code[0] for label, code in codes.items()}
        roster, _ = joined(base)
        banks = load_fits(base, np)
        args.output.mkdir(parents=True, exist_ok=False)
        result = {}
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            for backend in base["backends"]:
                result[backend] = evaluate(backend, base, cfg, head, roster, tokens, banks,
                                           args.output, check, np)
                print(backend, result[backend]["replayed_states"], flush=True)
        assert not any(key == "torch" or key.startswith("torch.") for key in sys.modules)
    write_new(args.output / "aggregate.json", {
        "experiment": cfg["experiment"], "status": "completed_prespecified_archive_replay",
        "source_manifest_sha256": digest(MANIFEST), "independent_roots_by_split": base["roots"],
        "formal_comparisons": 0, "new_pretrained_forwards": 0, "sealed_test_accessed": False,
        "verdict": "descriptive_observational", "per_backend": result,
        "resource": {"max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
                     "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples)},
    })


if __name__ == "__main__":
    main()
