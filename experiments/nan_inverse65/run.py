"""Bound archived residual operands and replay native head contrasts."""
import argparse
from collections import defaultdict
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_carry63.algebra import continuous, margins
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import FixedHead, SHARD, bound, digest, joined, write_new

from .lattice import invert, lattice, test


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
READPATH = REPO / "experiments/nan_readpath37"
ARCHIVE = REPO / "runs/nan_comp14"
SITES = ("residual_after_layer", "o_proj_output", "down_proj_output")
BOUNDS = ("incoming", "pre_mlp")
METRICS = ("finite_margin", "attention_write", "mlp_write", "rms_scale", "head_rounding",
           "effective_incoming", "effective_pre_mlp")


def seal():
    paths = [HERE / p for p in (".gitattributes", "PROTOCOL.md", "__init__.py", "registration.json",
                               "lattice.py", "run.py")]
    paths += [REPO / name for name in (
        "experiments/nan_readpath37/registration.json", "experiments/nan_readpath37/run.py",
        "experiments/nan_carry63/algebra.py", "experiments/nan_phase35/run.py",
        "experiments/nan_lattice53/run.py", "experiments/nan_ram28/resources.py",
        "experiments/nan_ranksite13/resources.py", "experiments/nan_premlp64/CPU_VERIFICATION.json")]
    return {"experiment": "NAN-INVERSE-65",
            "files_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in paths}}


def open_archive(backend, base, selected, head, stages, check, np):
    pin = base["backends"][backend]
    folder = ARCHIVE / backend
    audit = json.loads(bound(ARCHIVE / "audits" / f"{backend}.json", pin["audit"]))
    manifest = json.loads(bound(folder / "manifest.json", pin["manifest"]))
    if ((audit["status"], audit["worker_status"], audit["missing_contexts"], audit["contexts_checked"])
            != ("passed", "completed", 0, 240) or len(audit["records"]) != 240
            or manifest["checkpoint_revision"] != base["revision"]
            or manifest["target_parameter_sha256"] != pin["target"]):
        raise ValueError("Archive target or native audit changed")
    max_error = 0.0
    records = defaultdict(dict)
    expected = {(pass_index, layer, site) for pass_index, layer in stages for site in SITES}
    for slot, row in selected.items():
        check()
        verified = audit["records"][slot]
        case = json.loads(bound(folder / f"case_{slot:04}.json", verified["case_sha256"]))
        if any(row[k] != case[k] or row[k] != verified[k] for k in ("id", "root", "split", "arm", "world")):
            raise ValueError("Archive identity differs from pinned roster")
        state_name = f"case_{slot:04}_states.npz"
        if (case["state_file"], case["state_file_sha256"]) != (state_name, verified["state_sha256"]):
            raise ValueError("Archive state binding mismatch")
        entries = [x for x in case["state_index"] if x["site"] in SITES]
        index = {(x["loop"], x["layer_zero_based"], x["site"]): x["array_key"] for x in entries}
        if len(entries) != len(expected) or set(index) != expected:
            raise ValueError("Unexpected archive observation sites")
        with np.load(BytesIO(bound(folder / state_name, verified["state_sha256"])), allow_pickle=False) as data:
            stack = np.asarray([[data[index[phase, layer, name]] for name in SITES] for phase, layer in stages])
            if stack.dtype != np.float32 or stack.shape != (len(stages), len(SITES), base["width"]):
                raise ValueError("Archive numerical dtype or shape differs")
        if [x["loop"] for x in case["initial_phases"]] != [1, 2]:
            raise ValueError("Missing native phases")
        for item in case["initial_phases"]:
            score = item["score"]
            tokens = {int(k): v for k, v in score["selected_logits"].items()}
            tokens[score["free_top_token_id"]] = score["free_top_token_logit"]
            resid = stack[stages.index((item["loop"], 21)), 0]
            logit = head.logits(resid, list(tokens))
            max_error = max(max_error, float(np.max(abs(logit - list(tokens.values())))))
        group = records[row["split"], row["arm"], row["root"]]
        if row["world"] in group:
            raise ValueError("Repeated world in archive root")
        group[row["world"]] = (row["answer"], case["initial_prompt_ids_sha256"], stack)
    if max_error > base["native_logit_abs_tol"] or len(records) * 2 != len(selected):
        raise ValueError("Native replay or registered coverage differs")
    return records, max_error


def linear_interval(lower, upper, weights, np):
    # A/B numeric difference so the smallest B is paired with the largest A.
    smallest = lower[0] - upper[1]
    largest = upper[0] - lower[1]
    products = smallest * weights, largest * weights
    lo = np.minimum(*products)
    hi = np.maximum(*products)
    # Expand the FP64 reduction against cancellation and allow rounded products.
    slack = 1e-11 * (np.sum(abs(lo)) + np.sum(abs(hi))) + 1e-12
    return [float(np.sum(lo) - slack), float(np.sum(hi) + slack)]


def world_pair(stacked, grid, scale_direction, token_ids, head, erased, np):
    h, attention, mlp = (stacked[:, i, :] for i in range(3))
    a, b = margins(head, h, token_ids, np)
    finite = float(a - b)
    scaled, rms, continuous_swing = continuous(h.astype(np.float64), scale_direction, head.eps, np)
    signs = scaled, -scaled
    attention64, mlp64, h64 = (x.astype(np.float64) for x in (attention, mlp, h))
    eff_pre = h64 - mlp64
    eff_in = eff_pre - attention64
    terms = {
        "finite_margin": finite, "attention_write": float(np.sum(attention64 * signs)),
        "mlp_write": float(np.sum(mlp64 * signs)), "rms_scale": rms,
        "head_rounding": finite - continuous_swing,
        "effective_pre_mlp": float(np.sum(eff_pre * signs)),
        "effective_incoming": float(np.sum(eff_in * signs)),
    }
    closure = (terms["effective_incoming"] + terms["attention_write"] + terms["mlp_write"]
               + terms["rms_scale"] + terms["head_rounding"])
    if abs(closure - finite) > 1e-10:
        raise ValueError("Continuous incoming and write terms failed closure")

    lo, hi, carry_lo, carry_hi = invert(grid, h, attention, mlp, np)
    lengths = hi - lo + 1
    possible = lengths > 0
    counts = {"coordinates": int(lengths.size), "unreachable": int((~possible).sum()),
              "unique_incoming": int((lengths == 1).sum()),
              "unique_pre_mlp": int((possible & (carry_lo == carry_hi)).sum()),
              "max_preimages": int(lengths.max())}
    bounds = dict.fromkeys(BOUNDS)
    if possible.all():
        for name, low, high in (
                ("incoming", grid[lo], grid[hi]), ("pre_mlp", carry_lo, carry_hi)):
            bounds[name] = [0., 0.] if erased else linear_interval(
                low.astype(np.float64), high.astype(np.float64), scaled, np)
    if erased and (a != b or any(terms.values())):
        raise ValueError("Erased worlds differ under a fixed readout")
    return {"terms": terms, "bounds": bounds, "counts": counts,
            "both_bare_labels_win": bool(a > 0 and b < 0)}


def difference(a, b):
    return None if a is None or b is None else [b[0] - a[1], b[1] - a[0]]


def interval_cell(values):
    available = [x for x in values if x is not None]
    return {"roots_bounded": len(available), "roots_unbounded": len(values) - len(available),
            "roots_strictly_positive": sum(x[0] > 0 for x in available),
            "roots_strictly_negative": sum(x[1] < 0 for x in available),
            "max_interval_width": max((x[1] - x[0] for x in available), default=None),
            "mean_interval": [fmean(x[i] for x in available) for i in range(2)] if available else None}


def aggregate(groups, stages):
    result = {}
    for group, pairs in groups.items():
        sites, phase_gain = {}, {}
        for stage in stages:
            rows = [pair[stage] for pair in pairs]
            sites[f"pass{stage[0]}/layer{stage[1]}"] = {
                "mean_signed": {k: fmean(row["terms"][k] for row in rows) for k in METRICS},
                "intervals": {k: interval_cell([row["bounds"][k] for row in rows]) for k in BOUNDS},
                "counts": {k: sum(row["counts"][k] for row in rows) for k in
                           ("coordinates", "unreachable", "unique_incoming", "unique_pre_mlp")},
                "max_preimages": max(row["counts"]["max_preimages"] for row in rows),
                "both_bare_labels_win_roots": sum(row["both_bare_labels_win"] for row in rows),
            }
        for phase, layer in stages:
            if phase == 2:
                cells = [(pair[1, layer], pair[2, layer]) for pair in pairs]
                phase_gain[str(layer)] = {
                    "mean_signed_gain": {k: fmean(b["terms"][k] - a["terms"][k] for a, b in cells)
                                         for k in METRICS},
                    "intervals": {k: interval_cell([difference(a["bounds"][k], b["bounds"][k])
                                                   for a, b in cells]) for k in BOUNDS},
                }
        result["/".join(group)] = {"roots": len(pairs), "sites": sites, "phase2_minus_phase1": phase_gain}
    return result


def evaluate(backend, base, selected, head, labels, grid, stages, output, check, np):
    records, error = open_archive(backend, base, selected, head, stages, check, np)
    grouped = defaultdict(list)
    private = []
    for (split, arm, root), worlds in sorted(records.items()):
        check()
        assert set(worlds) == {"A", "B"}
        a, aid, sa = worlds["A"]
        b, bid, sb = worlds["B"]
        assert a != b
        erased = arm == "focused_erased"
        if erased and (aid != bid or not np.array_equal(sa, sb)):
            raise ValueError("Erased prompts or physical states differ")
        tokens = [labels[a], labels[b]]
        direction = head.gamma.astype(np.float64) * (head.row(tokens[0]).astype(np.float64) - head.row(tokens[1]))
        pair = {site: world_pair(np.stack((x, y)), grid, direction, tokens, head, erased, np)
                for site, x, y in zip(stages, sa, sb, strict=True)}
        grouped[split, arm].append(pair)
        private.append({"root": root, "split": split, "arm": arm,
                        "sites": {f"{p}/{l}": result for (p, l), result in pair.items()}})
    assert {k: len(v) for k, v in grouped.items()} == {
        (split, arm): count for split, count in base["roots"].items() for arm in base["arms"]}
    write_new(output / f"{backend}_private.json", private)
    return {"replayed_native_states": len(selected) * 2, "max_native_logit_error": error,
            "sampled_states": len(selected) * len(stages), "groups": aggregate(grouped, stages)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        cli.error("Enable Python assertions")
    if args.freeze:
        write_new(MANIFEST, seal())
        return
    if not args.snapshot or not args.output or json.loads(MANIFEST.read_bytes()) != seal():
        cli.error("Require pinned source, snapshot and output")
    cfg = json.loads((HERE / "registration.json").read_bytes())
    base = json.loads((READPATH / "registration.json").read_bytes())
    assert args.snapshot.name == cfg["revision"] == base["revision"]
    stages = [(p, l) for p in (1, 2) for l in cfg["layers"]]
    with lease(cfg) as (check, samples):
        assert (version("numpy"), version("tokenizers")) == ("2.5.5", "0.20.3")
        import numpy as np
        from tokenizers import Tokenizer

        cases_checked = test(np)
        grid = lattice(np)
        config = json.loads(bound(args.snapshot / "config.json", base["snapshot_pins"]["config.json"]))
        definitions = bound(args.snapshot / "configuration_nanbeige.py", cfg["configuration_sha256"])
        assert b"enable_hyper_connection=False," in definitions
        assert not config.get("enable_hyper_connection", False) and config["pretraining_tp"] == 1
        assert config["torch_dtype"] == "bfloat16"
        bound(args.snapshot / "tokenizer.json", base["snapshot_pins"]["tokenizer.json"])
        tok = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
        vocab = {label: tok.encode(label, add_special_tokens=False).ids for label in "ABCDEFGHIJKLMNOP"}
        assert all(len(code) == 1 for code in vocab.values()) and len({v[0] for v in vocab.values()}) == 16
        labels = {label: ids[0] for label, ids in vocab.items()}
        selected, _ = joined(base)
        result = {}
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.snapshot / SHARD).open("rb") as stream:
            head = FixedHead(stream, args.snapshot, base, np, check)
            for backend in base["backends"]:
                result[backend] = evaluate(backend, base, selected, head, labels, grid,
                                           stages, args.output, check, np)
                print(backend, result[backend]["sampled_states"], flush=True)
        assert not any(key == "torch" or key.startswith("torch.") for key in sys.modules)
    write_new(args.output / "aggregate.json", {
        "experiment": cfg["experiment"], "status": "completed", "source_manifest_sha256": digest(MANIFEST),
        "lattice_finite_numeric_values": len(grid), "exhaustive_synthetic_checks": cases_checked,
        "independent_roots": base["roots"], "formal_comparisons": 0, "new_pretrained_forwards": 0,
        "sealed_test_accessed": False, "per_backend": result,
        "resources": {"max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
                      "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples)},
    })


if __name__ == "__main__":
    main()
